import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';
import { ARTelemetry } from './ar_telemetry.js';

// The simulator owns every physical pose. Dressing adds context outside the work volume.
export class MineScene {
  constructor(container) {
    this.container = container;
    this.meshes = new Map();
    this.materials = new Map();
    this.labels = [];
    this.cartLabels = [];
    this.mode = 'overview';
    this.showSurvey = new URLSearchParams(window.location.search).get('survey') === '1';
    container.style.position = 'relative';
    try {
      this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'high-performance' });
    } catch (error) {
      container.innerHTML = '<div style="padding:40px;color:#edc88b">The live excavation view needs WebGL. Enable browser hardware acceleration and reload.</div>';
      this.error = error;
      return;
    }
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.35;
    this.renderer.domElement.style.cssText = 'display:block;width:100%;height:100%;touch-action:none';
    container.appendChild(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color('#171b20');
    this.scene.fog = new THREE.Fog('#171b20', 3.5, 8);
    this.camera = new THREE.PerspectiveCamera(38, 1, 0.008, 30);
    this.camera.up.set(0, 0, 1);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.075;
    this.controls.minDistance = 0.25;
    this.controls.maxDistance = 4.5;
    this.controls.maxPolarAngle = Math.PI * 0.47;
    this.controls.screenSpacePanning = true;
    this.physics = new THREE.Group();
    this.scene.add(this.physics);
    this.rotation = new THREE.Matrix4();
    this.projected = new THREE.Vector3();
    this.loader = new THREE.TextureLoader();
    this.textures = {};
    for (const [key, folder, file] of [
      ['soil', 'soil_ambientcg_Ground112', 'Ground112_1K-JPG_Color.jpg'],
      ['soilNormal', 'soil_ambientcg_Ground112', 'Ground112_1K-JPG_NormalGL.jpg'],
      ['rock', 'rock_ambientcg_Rock064', 'Rock064_1K-JPG_Color.jpg'],
      ['rockNormal', 'rock_ambientcg_Rock064', 'Rock064_1K-JPG_NormalGL.jpg'],
    ]) {
      const texture = this.loader.load(`/assets/${folder}/${file}`);
      texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
      texture.repeat.set(key.startsWith('soil') ? 3 : 1.5, key.startsWith('soil') ? 3 : 1.5);
      texture.anisotropy = Math.min(8, this.renderer.capabilities.getMaxAnisotropy());
      if (!key.endsWith('Normal')) texture.colorSpace = THREE.SRGBColorSpace;
      this.textures[key] = texture;
    }
    this.buildLights();
    this.buildSite();
    this.buildQuarry();
    this.forceAnchor = new THREE.Vector3();
    this.forceArrow = new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), this.forceAnchor, 0.1, 0x58d7d1, 0.025, 0.012);
    this.forceArrow.visible = false;
    this.scene.add(this.forceArrow);
    this.gripArrows = [];
    this.ar = new ARTelemetry(this.scene);
    this.poseTransitions = new Map();
    this.frameTimes = [];
    this.lastRenderAt = performance.now();
    this.buildWristLamp();
    this.loadSubterraneanEnvironment();
    this.loadEnvironment();
    this.setupPostprocessing();
    this.cameraMode('overview');
    this.observer = new ResizeObserver(() => this.resize());
    this.observer.observe(container);
    this.resize();
    this.renderer.setAnimationLoop(() => {
      const now = performance.now();
      this.frameTimes.push(now-this.lastRenderAt);this.lastRenderAt=now;
      if(this.frameTimes.length>600)this.frameTimes.shift();
      for(const [mesh,transition] of this.poseTransitions) {
        const alpha=Math.min(1,(now-transition.at)/50);
        mesh.position.lerpVectors(transition.from,transition.to,alpha);
        mesh.quaternion.slerpQuaternions(transition.fromQ,transition.toQ,alpha);
        if(alpha>=1)this.poseTransitions.delete(mesh);
      }
      this.ar.tick(now);
      this.updateChaseCamera();
      this.controls.update();
      if (this.composer) {
        try {
          this.composer.render();
        } catch (error) {
          console.warn('Postprocessing unavailable; using direct rendering.', error);
          this.disposePostprocessing();
          this.renderer.setRenderTarget(null);
          this.renderer.render(this.scene, this.camera);
        }
      } else {
        this.renderer.render(this.scene, this.camera);
      }
      this.updateLabels();
      window.__deepcoreRenderStats={samples:this.frameTimes.length,
        mean_frame_ms:this.frameTimes.reduce((a,b)=>a+b,0)/this.frameTimes.length,
        fps:1000/(this.frameTimes.reduce((a,b)=>a+b,0)/this.frameTimes.length),
        renderer:'Three.js WebGL',pose_source:'Measured MuJoCo poses interpolated over 50 ms'};
    });
  }

  async loadEnvironment() {
    let hdr, generator;
    try {
      const { RGBELoader } = await import('./vendor/addons/loaders/RGBELoader.js');
      if (this.disposed) return;
      hdr = await new RGBELoader().loadAsync('/assets/hdri_polyhaven/kloofendal_48d_partly_cloudy_puresky_1k.hdr');
      if (this.disposed) return;
      generator = new THREE.PMREMGenerator(this.renderer);
      this.environmentTarget = generator.fromEquirectangular(hdr);
      this.scene.environment = this.environmentTarget.texture;
      this.scene.environmentIntensity = 0.3;
      // The sky lights the materials; the existing charcoal background stays visible.
    } catch (error) {
      console.warn('HDR lighting unavailable; retaining site lights.', error);
    } finally {
      hdr?.dispose();
      generator?.dispose();
    }
  }

  async setupPostprocessing() {
    try {
      const [{ EffectComposer }, { RenderPass }, { UnrealBloomPass }, { OutputPass }] = await Promise.all([
        import('./vendor/addons/postprocessing/EffectComposer.js'),
        import('./vendor/addons/postprocessing/RenderPass.js'),
        import('./vendor/addons/postprocessing/UnrealBloomPass.js'),
        import('./vendor/addons/postprocessing/OutputPass.js'),
      ]);
      if (this.disposed) return;
      this.composer = new EffectComposer(this.renderer);
      this.composer.addPass(new RenderPass(this.scene, this.camera));
      // Only the high luminance lamp faces contribute appreciably; preserve core detail.
      this.composer.addPass(new UnrealBloomPass(new THREE.Vector2(1, 1), 0.14, 0.22, 2.5));
      this.composer.addPass(new OutputPass());
      this.resize();
    } catch (error) {
      console.warn('Postprocessing unavailable; using direct rendering.', error);
      this.disposePostprocessing();
    }
  }

  disposePostprocessing() {
    if (!this.composer) return;
    for (const pass of this.composer.passes) pass.dispose?.();
    this.composer.dispose();
    this.composer = null;
  }

  async loadSubterraneanEnvironment() {
    try {
      const {loadMineEnvironment}=await import('/assets/mine_environment/environment.js');
      if(this.disposed)return;
      this.mineEnvironment=await loadMineEnvironment(this.scene);
    }catch(error) {
      this.container.dispatchEvent(new CustomEvent('mine-load-error',{detail:error.message}));
    }
  }

  buildWristLamp() {
    this.wristLamp=new THREE.SpotLight(0xd6eaff,5,2.5,Math.PI/7,.65,1.3);
    this.wristLamp.castShadow=true;this.wristLamp.shadow.mapSize.set(1024,1024);
    this.wristLamp.shadow.bias=-.0001;this.wristLamp.shadow.normalBias=.002;
    this.scene.add(this.wristLamp,this.wristLamp.target);
    this.lightCone=new THREE.Mesh(new THREE.ConeGeometry(1,1,40,1,true),new THREE.MeshBasicMaterial({
      color:0xc2eaff,transparent:true,opacity:.025,side:THREE.DoubleSide,depthWrite:false,blending:THREE.AdditiveBlending}));
    this.lightCone.rotation.x=Math.PI/2;this.scene.add(this.lightCone);
    this.coreGlow=new THREE.PointLight(0x22e9e5,.12,.30,2);this.scene.add(this.coreGlow);
  }

  updateWristLamp(frame) {
    const position=frame.fleet?.collector_tcp_m||frame.ee;
    if(!position?.every(Number.isFinite))return;
    this.wristLamp.position.fromArray(position).add(new THREE.Vector3(.018,-.022,.055));
    this.wristLamp.target.position.copy(this.wristLamp.position).add(new THREE.Vector3(0,.03,-1));
    const length=Math.max(.07,this.wristLamp.position.z-.40),radius=length*Math.tan(Math.PI/7);
    this.lightCone.scale.set(radius,length,radius);
    this.lightCone.position.copy(this.wristLamp.position).add(new THREE.Vector3(0,0,-length/2));
    this.coreGlow.visible=Boolean(this.coreMesh);
    if(this.coreMesh)this.coreGlow.position.copy(this.coreMesh.position).add(new THREE.Vector3(0,0,.035));
  }

  updateChaseCamera(instant=false) {
    if(this.mode!=='chase'||!this.lastFrame)return;
    const wrist=new THREE.Vector3(...(this.lastFrame.fleet?.collector_tcp_m||this.lastFrame.ee||[.42,0,.7]));
    const offset=this.fleet?new THREE.Vector3(.65,-.85,.30):new THREE.Vector3(.34,-.43,.21);
    const destination=wrist.clone().add(offset);
    this.camera.position.lerp(destination,instant?1:.12);
    const target=wrist.clone().add(new THREE.Vector3(-.07,.06,-.12));
    this.controls.target.lerp(target,instant?1:.12);this.camera.lookAt(this.controls.target);
  }

  buildLights() {
    this.scene.add(new THREE.HemisphereLight(0xa5becb, 0x151616, .6));
    const sun = new THREE.DirectionalLight(0xc6dce7, 1.35);
    sun.position.set(0.2, -1.3, 3.6);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    Object.assign(sun.shadow.camera, { left: -1.8, right: 1.8, top: 1.8, bottom: -1.8, near: 0.1, far: 7 });
    sun.shadow.normalBias = 0.0015;
    sun.shadow.bias = -0.00015;
    sun.shadow.radius = 3;
    sun.target.position.set(0.45, 0, 0.4);
    this.scene.add(sun, sun.target);
    const rim = new THREE.DirectionalLight(0x98cbea, .65);
    rim.position.set(-1.4, 1.5, 2);
    this.scene.add(rim);
  }

  material(geom) {
    const key = geom.material || `rgba-${geom.rgba?.join(',')}`;
    if (this.materials.has(key)) return this.materials.get(key);
    const rgba = geom.rgba || [0.6, 0.6, 0.6, 1];
    const options = { color: new THREE.Color(...rgba.slice(0, 3)), roughness: 0.85, metalness: 0 };
    if (/earth|wall|rim/.test(key)) Object.assign(options, { color: key === 'wall' ? 0x535c60 : 0x646b69, map: this.textures.soil, normalMap: this.textures.soilNormal, normalScale: new THREE.Vector2(0.65, 0.65) });
    if (/rock|gravel|rubble/.test(key)) Object.assign(options, { map: this.textures.rock, normalMap: this.textures.rockNormal, normalScale: new THREE.Vector2(0.7, 0.7), roughness: 0.95 });
    if (/metal|steel/.test(key)) Object.assign(options, { color: 0xaebac2, metalness: 0.7, roughness: 0.31 });
    if (/ur5e_linkgray/.test(key)) Object.assign(options, {metalness:.72, roughness:.29});
    if (/ur5e_jointgray/.test(key)) Object.assign(options, {metalness:.52, roughness:.36});
    if (/ur5e_urblue/.test(key)) Object.assign(options, {metalness:.22, roughness:.33});
    if (/ur5e_black|^dark$/.test(key)) Object.assign(options, {metalness:.08, roughness:.76});
    if (/^paint$|collector_teal/.test(key)) Object.assign(options, {metalness:.3, roughness:.38});
    if (key === 'hazard') Object.assign(options, { color: 0xf3ad30, metalness: 0.25, roughness: 0.5 });
    if (key === 'pad') Object.assign(options, { color: 0x20292e, roughness: 0.95 });
    if (/core/.test(key)) Object.assign(options, { color: 0x69f0e4, metalness: 0.38, roughness: 0.2, emissive: 0x14d5d0, emissiveIntensity: 1.4, flatShading:true });
    if (/mat_rubble|debris_material/.test(key)) Object.assign(options,{color:0x6f7473,roughness:.96,metalness:.04,flatShading:true});
    const material = new THREE.MeshStandardMaterial(options);
    this.materials.set(key, material);
    return material;
  }

  geometry(geom) {
    const [x, y, z] = geom.size;
    let geometry;
    // Small bevels catch practical lights on manufactured housings. These are
    // display-only meshes; dimensions and measured rigid transforms are preserved.
    if (geom.type === 6 && /mobile_base($|_deck|_battery)|mobile_drill_(mast|arm)/.test(geom.name)) {
      const radius=Math.min(x,y,z)*.18;
      const shape=new THREE.Shape();
      shape.moveTo(-x+radius,-y+radius);shape.lineTo(x-radius,-y+radius);
      shape.lineTo(x-radius,y-radius);shape.lineTo(-x+radius,y-radius);shape.closePath();
      geometry=new THREE.ExtrudeGeometry(shape,{depth:2*(z-radius),bevelEnabled:true,
        bevelThickness:radius,bevelSize:radius,bevelSegments:3,steps:1,curveSegments:1});
      geometry.translate(0,0,-z+radius);geometry.computeVertexNormals();return geometry;
    }
    if (/^(rubble_[ab]|rock_base|boulder\d+|scree_\d+|debris_\d+)$/.test(geom.name)) {
      geometry=new THREE.IcosahedronGeometry(1,2);
      const position=geometry.attributes.position;
      const seed=[...geom.name].reduce((a,c)=>a+c.charCodeAt(0),0);
      for(let i=0;i<position.count;i++) {
        const a=position.getX(i),b=position.getY(i),c=position.getZ(i);
        const noise=.86+.13*Math.sin(a*17.1+b*21.7+c*13.4+seed);
        const shape=v=>Math.sign(v)*Math.pow(Math.abs(v),.55)*noise;
        position.setXYZ(i,shape(a)*x,shape(b)*y,shape(c)*z);
      }
      geometry.computeVertexNormals();return geometry;
    }
    if(geom.name==='core') {
      geometry=new THREE.DodecahedronGeometry(1,0);
      geometry.scale(x,y,z);geometry.computeVertexNormals();return geometry;
    }
    switch (geom.type) {
      case 7:
        if (geom.mesh?.vertices?.length) {
          geometry = new THREE.BufferGeometry();
          geometry.setAttribute('position', new THREE.Float32BufferAttribute(geom.mesh.vertices, 3));
          geometry.setIndex(geom.mesh.indices);
          geometry.computeVertexNormals();
          return geometry;
        }
        return new THREE.BoxGeometry(x * 2, y * 2, z * 2);
      case 0: return new THREE.PlaneGeometry(x * 2 || 6, y * 2 || 6);
      case 2: return new THREE.SphereGeometry(x, 24, 16);
      case 3: geometry = new THREE.CapsuleGeometry(x, y * 2, 8, 16); geometry.rotateX(Math.PI / 2); return geometry;
      case 4: geometry = new THREE.SphereGeometry(1, 16, 12); geometry.scale(x, y, z); return geometry;
      case 5: geometry = new THREE.CylinderGeometry(x, x, y * 2, 24); geometry.rotateX(Math.PI / 2); return geometry;
      default: return new THREE.BoxGeometry(x * 2, y * 2, z * 2);
    }
  }

  pose(mesh, pose) {
    if (pose.pos) mesh.position.fromArray(pose.pos);
    const m = pose.mat;
    if (m?.length === 9) {
      this.rotation.set(m[0], m[1], m[2], 0, m[3], m[4], m[5], 0, m[6], m[7], m[8], 0, 0, 0, 0, 1);
      mesh.quaternion.setFromRotationMatrix(this.rotation);
    }
  }

  load(geoms) {
    if (this.error || !Array.isArray(geoms)) return;
    for (const child of [...this.physics.children]) {
      child.geometry?.dispose();
      this.physics.remove(child);
    }
    this.meshes.clear();
    this.poseTransitions.clear();this.ar.reset();this.coreMesh=null;
    for (const label of this.labels) label.element.remove();
    this.labels = [];
    this.cartLabels = [];
    const wasFleet = this.fleet;
    this.fleet = geoms.some(geom => geom.name === 'drill_mobile_base');
    for (const geom of geoms) {
      // The surveyed triangle mesh is loaded directly, preserving concavity.
      if (geom.name.startsWith('cmu_') || geom.group >= 3) continue;
      const mesh = new THREE.Mesh(this.geometry(geom), this.material(geom));
      mesh.name = geom.name;
      mesh.castShadow = geom.type !== 0 && !/rim_|pit_floor/.test(geom.name);
      mesh.receiveShadow = true;
      this.pose(mesh, geom);
      this.physics.add(mesh);
      this.meshes.set(geom.id, mesh);
      if (/^(drill|collector)_mobile_base$/.test(geom.name)) this.addChassisDetails(mesh);
      if (geom.name === 'core') {
        this.coreMesh=mesh;this.addLabel('FRAGILE CORE', mesh, '#58e2e5', 0.045);
        for(let i=0;i<5;i++) {
          const shard=new THREE.Mesh(new THREE.ConeGeometry(.009,.027,5),mesh.material);
          shard.rotation.x=Math.PI/2;shard.rotation.z=i*1.23;shard.rotation.y=(i-2)*.22;
          shard.position.set(Math.cos(i*1.23)*.009,Math.sin(i*1.23)*.009,.009);
          mesh.add(shard);
        }
      }
      if (geom.name === 'drill_mobile_base') this.addLabel('01 · DRILLER', mesh, '#edb563', 0.39);
      if (geom.name === 'collector_mobile_base') this.addLabel('02 · COLLECTOR', mesh, '#58e2e5', 0.39);
      const cart = /^cart_(\d+)_floor$/.exec(geom.name);
      if (cart) this.cartLabels[Number(cart[1])] = this.addLabel(`CART ${Number(cart[1]) + 1}`, mesh, '#c6d6a3', .22);
    }
    const drilling = geoms.some(geom => geom.name === 'cutter_0');
    this.siteDressing.visible = !drilling && !this.fleet;
    this.quarry.visible = false;
    this.quarry.position.z = drilling ? -0.225 : 0;
    if (!drilling && !this.fleet) this.addLabel('RECOVERY TRAY', new THREE.Vector3(0.52, 0.16, 0.435), '#c6d6a3', 0.025);
    if (this.fleet) {
      this.loadMine();
      this.drillLoadAnchor = new THREE.Vector3();
      this.gripLoadAnchor = new THREE.Vector3();
      this.drillLoadLabel = this.addLabel('', this.drillLoadAnchor, '#edb563', .07);
      this.gripLoadLabel = this.addLabel('', this.gripLoadAnchor, '#58e2e5', .04);
    }
    if (this.cmuMine) this.cmuMine.group.visible = this.fleet && this.showSurvey;
    this.controls.maxDistance = this.fleet ? 12 : 4.5;
    this.scene.fog = new THREE.FogExp2('#0b1217', .075);
    this.scene.background = new THREE.Color('#0b1217');
    if (wasFleet !== this.fleet) this.cameraMode('overview');
    this.forceLabel = this.addLabel('', this.forceAnchor, '#58d7d1', 0.04);
    this.forceLabel.visible = false;
    for (const arrow of this.gripArrows) arrow.visible = false;
  }

  async loadMine() {
    if (this.cmuMine || this.mineLoading) return;
    this.mineLoading = true;
    try {
      const { loadCMUMine } = await import('./cmu_mine.js');
      const mine = await loadCMUMine(this.scene);
      if (!mine.metadata) throw mine.error || new Error('CMU mesh assets are missing');
      if (this.disposed) { mine.dispose(); return; }
      this.cmuMine = mine;
      mine.group.visible = Boolean(this.fleet && this.showSurvey);
    } catch (error) {
      console.error('CMU mine could not be loaded.', error);
      this.container.dispatchEvent(new CustomEvent('mine-load-error', {detail: error.message}));
    } finally { this.mineLoading = false; }
  }

  addChassisDetails(chassis) {
    // Fasteners and guards follow the measured chassis transform. They are
    // presentation children, never controller observations or collision shapes.
    const steel = new THREE.MeshStandardMaterial({color:0xb9c2c8,metalness:.8,roughness:.32});
    const rubber = new THREE.MeshStandardMaterial({color:0x141b20,roughness:.86});
    const amber = new THREE.MeshStandardMaterial({color:0xf7bd54,metalness:.15,roughness:.45});
    for (const x of [-.155,.155]) for (const y of [-.093,.093]) {
      const bolt=new THREE.Mesh(new THREE.CylinderGeometry(.006,.006,.004,6),steel);
      bolt.rotation.x=Math.PI/2;bolt.position.set(x,y,.047);chassis.add(bolt);
    }
    for (const x of [-.181,.181]) {
      const bumper=new THREE.Mesh(new THREE.BoxGeometry(.012,.20,.026),rubber);
      bumper.position.set(x,0,-.005);chassis.add(bumper);
      for(const y of [-.079,.079]) {
        const reflector=new THREE.Mesh(new THREE.BoxGeometry(.013,.028,.008),amber);
        reflector.position.set(x,y,.014);chassis.add(reflector);
      }
    }
  }

  update(frame) {
    if (this.error || !frame) return;
    for (const pose of frame.poses || []) {
      const mesh = this.meshes.get(pose.id);
      if (mesh) {
        const from=mesh.position.clone(),fromQ=mesh.quaternion.clone();this.pose(mesh,pose);
        this.poseTransitions.set(mesh,{from,fromQ,to:mesh.position.clone(),toQ:mesh.quaternion.clone(),at:performance.now()});
        mesh.position.copy(from);mesh.quaternion.copy(fromQ);
        if ('visible' in pose) mesh.visible = pose.visible;
      }
    }
    this.lastFrame=frame;this.ar.update(frame);this.updateWristLamp(frame);
    // This bridge and carriage follow the measured Cartesian axes exactly.
    if (frame.ee?.length >= 3) {
      this.bridge.position.x = frame.ee[0];
      this.carriage.position.set(frame.ee[0], frame.ee[1], 1.045);
    }
    const wrench = frame.drilling?.contact_force_world_n || frame.wrist_vector;
    const vector = new THREE.Vector3(...(wrench || [0, 0, 0]));
    const magnitude = vector.length();
    this.forceArrow.visible = magnitude > 0.5;
    if (this.forceLabel) this.forceLabel.visible = magnitude > 0.5;
    if (magnitude > 0.5) {
      this.forceAnchor.fromArray(frame.drilling?.contact_position_m || frame.ee || [0.42, 0, 0.45]);
      this.forceArrow.position.copy(this.forceAnchor);
      this.forceArrow.setDirection(vector.normalize());
      const limit = frame.drilling?.limit_force_n || 28;
      this.forceArrow.setLength(0.06 + Math.min(1, magnitude / limit) * 0.18, 0.022, 0.012);
      this.forceArrow.setColor(frame.drilling?(magnitude>limit?0xff365b:0x58d7d1):magnitude>18?0xff365b:magnitude>=10?0xf6b44d:0x58d7d1);
      if (this.forceLabel) this.forceLabel.element.textContent = `${magnitude.toFixed(1)} N · TOOL LOAD`;
    }
    const fleet = frame.fleet;
    if (fleet && this.gripLoadLabel) {
      this.drillLoadAnchor.fromArray(frame.ee || [0, 0, 0]);
      this.gripLoadAnchor.fromArray(fleet.collector_tcp_m || fleet.machine_positions?.collector || [0, 0, 0]);
      this.drillLoadLabel.element.textContent = `THRUST ${(fleet.axial_force_n || 0).toFixed(1)} N · TORQUE ${(fleet.torque_nm || 0).toFixed(2)} N·m`;
      this.gripLoadLabel.element.textContent = `GRIP CONTACT ${(fleet.collector_contact_force_n || 0).toFixed(1)} N`;
      this.cartLabels.forEach((label, i) => {
        const count = fleet.cart_counts?.[i] ?? 0;
        const travel = fleet.cart_travel_m?.[i] ?? 0;
        const status = fleet.cart_departed?.[i] ? 'DISPATCHED' : travel < -.03 ? 'DEPARTING' : 'LOADING';
        label.element.textContent = `CART ${i + 1} · ${count}/${fleet.cart_capacity ?? 2} · ${status}`;
      });
      if (this.forceLabel) this.forceLabel.visible = false;
    }
    const contacts = [...(fleet?.collector_contacts || [])].sort((a, b) =>
      Math.hypot(...b.force_world_n) - Math.hypot(...a.force_world_n)).slice(0, 2);
    for (let i = 0; i < Math.max(contacts.length, this.gripArrows.length); i++) {
      if (!this.gripArrows[i]) {
        const arrow = new THREE.ArrowHelper(new THREE.Vector3(0, 1, 0), new THREE.Vector3(), .1, 0x58d7d1, .018, .009);
        this.gripArrows.push(arrow); this.scene.add(arrow);
      }
      const arrow = this.gripArrows[i], contact = contacts[i];
      arrow.visible = false;
      if (!contact) continue;
      const direction = new THREE.Vector3(...contact.force_world_n);
      const force = direction.length();
      if (force < .1) continue;
      arrow.visible = true;
      arrow.position.fromArray(contact.position_m);
      arrow.setDirection(direction.normalize());
      arrow.setLength(.035 + Math.min(force / 15, 1) * .1, .012, .007);
    }
  }

  addLabel(text, anchor, color, offset) {
    const element = document.createElement('div');
    element.textContent = text;
    element.style.cssText = `position:absolute;left:0;top:0;pointer-events:none;color:${color};font:600 9px/1.3 ui-monospace,SFMono-Regular,monospace;letter-spacing:1.3px;padding:5px 7px;border-left:2px solid ${color};background:rgba(13,20,25,.73);white-space:nowrap;transform:translate(-999px,-999px)`;
    this.container.appendChild(element);
    const label = { element, anchor, offset, visible: true };
    this.labels.push(label);
    return label;
  }

  updateLabels() {
    const width = this.container.clientWidth, height = this.container.clientHeight;
    for (const label of this.labels) {
      if (label.visible === false) { label.element.style.display = 'none'; continue; }
      this.projected.copy(label.anchor.isVector3 ? label.anchor : label.anchor.position);
      this.projected.z += label.offset;
      this.projected.project(this.camera);
      const x = (this.projected.x * 0.5 + 0.5) * width;
      const y = (-this.projected.y * 0.5 + 0.5) * height;
      label.element.style.display = this.projected.z > 1 || x < 0 || x > width || y < 0 || y > height ? 'none' : 'block';
      label.element.style.transform = `translate(${Math.round(x + 9)}px,${Math.round(y - 20)}px)`;
    }
  }

  buildSite() {
    const site = new THREE.Group();
    this.siteDressing = site;
    this.scene.add(site);
    const amber = new THREE.MeshStandardMaterial({ color: 0xd79124, roughness: 0.48, metalness: 0.55 });
    const dark = new THREE.MeshStandardMaterial({ color: 0x252e31, roughness: 0.6, metalness: 0.6 });
    const steel = new THREE.MeshStandardMaterial({ color: 0x93a0a2, roughness: 0.32, metalness: 0.8 });
    const lamp = new THREE.MeshStandardMaterial({ color: 0xffe5b7, emissive: 0xffd494, emissiveIntensity: 3 });
    const box = (size, position, material = dark, parent = site) => {
      const mesh = new THREE.Mesh(new THREE.BoxGeometry(...size), material);
      mesh.position.set(...position); mesh.castShadow = mesh.receiveShadow = true; parent.add(mesh); return mesh;
    };
    const beam = (a, b, radius, material = steel) => {
      const start = new THREE.Vector3(...a), end = new THREE.Vector3(...b);
      const mesh = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, start.distanceTo(end), 10), material);
      mesh.position.copy(start).add(end).multiplyScalar(0.5);
      mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), end.sub(start).normalize());
      mesh.castShadow = mesh.receiveShadow = true; site.add(mesh); return mesh;
    };
    // Small industrial portal, outside the basin and above the real robot stroke.
    for (const x of [-0.03, 0.93]) for (const y of [-0.56, 0.56]) {
      box([0.085, 0.085, 0.025], [x, y, 0.637], steel);
      box([0.035, 0.035, 0.46], [x, y, 0.88], amber);
      box([0.041, 0.041, 0.055], [x, y, 0.69]);
      for (const dx of [-0.027, 0.027]) for (const dy of [-0.027, 0.027]) box([0.008, 0.008, 0.006], [x + dx, y + dy, 0.654], dark);
    }
    for (const y of [-0.56, 0.56]) {
      box([1.07, 0.045, 0.055], [0.45, y, 1.09], amber);
      box([1.07, 0.015, 0.009], [0.45, y, 1.123], steel);
    }
    this.bridge = new THREE.Group(); this.bridge.position.x = 0.42; site.add(this.bridge);
    box([0.034, 1.17, 0.035], [0, 0, 1.065], dark, this.bridge);
    box([0.012, 1.13, 0.009], [0, 0, 1.088], steel, this.bridge);
    this.carriage = new THREE.Group(); this.carriage.position.set(0.42, 0, 1.045); site.add(this.carriage);
    box([0.065, 0.075, 0.037], [0, 0, 0], amber, this.carriage);
    box([0.035, 0.055, 0.007], [0, 0, 0.024], dark, this.carriage);
    // Subtle diagonal bracing on the back side keeps the working area open.
    beam([-0.03, 0.56, 0.69], [0.13, 0.56, 1.07], 0.009, amber);
    beam([0.93, 0.56, 0.69], [0.77, 0.56, 1.07], 0.009, amber);
    // Rim edge strips are flush with the ground, never inside the collision volume.
    for (const y of [-0.505, 0.505]) {
      box([0.82, 0.025, 0.006], [0.45, y, 0.625], dark);
      for (let i = 0; i < 20; i++) {
        const stripe = box([0.022, 0.024, 0.007], [0.06 + i * 0.04, y, 0.626], amber);
        stripe.rotation.z = -0.35;
      }
    }
    // Floodlight masts, cable conduit and a compact site power unit.
    for (const [x, y] of [[-0.19, 0.68], [1.08, 0.68]]) {
      box([0.10, 0.10, 0.025], [x, y, 0.638], dark);
      beam([x, y, 0.65], [x, y, 1.05], 0.009, dark);
      box([0.085, 0.03, 0.046], [x, y, 1.06], dark);
      box([0.07, 0.003, 0.03], [x, y - 0.018, 1.06], lamp);
      const light = new THREE.SpotLight(0xffe3b5, 2.5, 2, 0.65, 0.7, 2);
      light.position.set(x, y, 1.06); light.target.position.set(0.45, 0, 0.4); site.add(light, light.target);
    }
    box([0.17, 0.11, 0.17], [-0.24, 0.22, 0.71], dark);
    box([0.175, 0.115, 0.016], [-0.24, 0.22, 0.8], amber);
    for (let i = 0; i < 6; i++) box([0.105, 0.003, 0.006], [-0.24, 0.163, 0.66 + i * 0.016], steel);
    const cableCurve = new THREE.CatmullRomCurve3([new THREE.Vector3(-0.24, 0.28, 0.64), new THREE.Vector3(-0.2, 0.53, 0.638), new THREE.Vector3(-0.03, 0.61, 0.638), new THREE.Vector3(-0.03, 0.56, 0.9)]);
    site.add(new THREE.Mesh(new THREE.TubeGeometry(cableCurve, 40, 0.007, 8, false), dark));
    // Deterministic, entirely exterior scree gives the cut earth a natural silhouette.
    let seed = 4761;
    const random = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };
    const rock = new THREE.MeshStandardMaterial({ color: 0x998d7c, map: this.textures.rock, normalMap: this.textures.rockNormal, roughness: 1 });
    const geometry = new THREE.IcosahedronGeometry(1, 1);
    const stones = new THREE.InstancedMesh(geometry, rock, 65);
    const dummy = new THREE.Object3D();
    for (let i = 0; i < 65; i++) {
      let x, y;
      do { x = -0.85 + random() * 2.65; y = -1.3 + random() * 2.6; } while (x > -0.31 && x < 1.19 && y > -0.72 && y < 0.8);
      const r = 0.012 + random() * 0.055;
      dummy.position.set(x, y, 0.625 + r * 0.15);
      dummy.scale.set(r, r * (0.65 + random()), r * (0.3 + random() * 0.6));
      dummy.rotation.set(random() * 3, random() * 3, random() * 6);
      dummy.updateMatrix(); stones.setMatrixAt(i, dummy.matrix);
    }
    stones.castShadow = stones.receiveShadow = true; site.add(stones);
  }

  buildQuarry() {
    // Render-only cut ground outside the working volume. Contacts remain owned by MuJoCo.
    this.quarry = new THREE.Group();
    const terrain = new THREE.PlaneGeometry(8, 8, 96, 96);
    const p = terrain.attributes.position;
    for (let i = 0; i < p.count; i++) {
      const x = p.getX(i) + 0.45, y = p.getY(i);
      const outside = Math.max(Math.abs(x - 0.45) / 1.45, Math.abs(y) / 1.2);
      const ridge = Math.max(0, y - 0.85) * 0.42 + Math.max(0, Math.abs(x - 0.45) - 1.6) * 0.4;
      const bench = Math.floor(ridge / 0.19) * 0.14 + ridge * 0.26;
      const rough = (Math.sin(x * 13 + y * 5) * Math.cos(y * 17) + Math.sin(x * 31 - y * 9) * 0.3) * 0.018;
      p.setXYZ(i, x, y, outside < 1 ? 0.35 : 0.623 + bench + rough);
    }
    terrain.computeVertexNormals();
    const material = new THREE.MeshStandardMaterial({color:0x8e8170, map:this.textures.soil,
      normalMap:this.textures.soilNormal, normalScale:new THREE.Vector2(0.8,0.8), roughness:1});
    const mesh = new THREE.Mesh(terrain, material);
    mesh.receiveShadow = true; this.quarry.add(mesh); this.scene.add(this.quarry);
  }

  cameraMode(mode) {
    if (this.error) return;
    this.mode = mode;
    this.controls.enabled=mode!=='chase';
    if(mode==='chase') {this.updateChaseCamera(true);return;}
    const poses = this.fleet ? {
      overview: [[2.6, -3.3, 1.72], [.15, .30, .78]],
      close: [[1.7, -2.0, 1.65], [.42, 0, .68]],
      top: [[.4, -.001, 4.6], [.4, 0, .4]],
    } : {
      overview: [[.68, -1.25, 1.12], [0.44, 0.02, 0.59]],
      close: [[0.92, -0.78, 1.14], [0.43, 0, 0.47]],
      top: [[0.45, -0.001, 2.3], [0.45, 0, 0.43]],
    };
    const [position, target] = poses[mode] || poses.overview;
    this.camera.position.fromArray(position); this.controls.target.fromArray(target);
    this.controls.update();
  }

  resize() {
    if (this.error) return;
    const width = Math.max(1, this.container.clientWidth), height = Math.max(1, this.container.clientHeight);
    this.camera.aspect = width / height; this.camera.updateProjectionMatrix(); this.renderer.setSize(width, height, false);
    if (this.composer) {
      this.composer.setPixelRatio(this.renderer.getPixelRatio());
      this.composer.setSize(width, height);
    }
  }

  dispose() {
    this.disposed = true;
    this.observer?.disconnect(); this.controls?.dispose();
    this.renderer?.setAnimationLoop(null);
    this.disposePostprocessing();
    if (this.scene) this.scene.environment = null;
    this.environmentTarget?.dispose();
    this.cmuMine?.dispose();
    for (const arrow of this.gripArrows || []) arrow.dispose();
    this.scene?.traverse(object => object.geometry?.dispose());
    for (const material of this.materials.values()) material.dispose();
    for (const texture of Object.values(this.textures || {})) texture.dispose();
    for (const label of this.labels) label.element.remove();
    this.renderer?.dispose(); this.renderer?.domElement.remove();
  }
}
