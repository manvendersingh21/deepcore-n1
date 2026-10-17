import * as THREE from 'three';

const valid = p => Array.isArray(p) && p.length === 3 && p.every(Number.isFinite);
const colors = {core:0x53f4eb, rubble_a:0xf2b86c, rubble_b:0xba9bff};

// All positions below originate in observed RGB-D, policy estimates or commands.
export class ARTelemetry {
  constructor(scene) {
    this.group = new THREE.Group(); scene.add(this.group);
    this.cloudGeometry = new THREE.BufferGeometry();
    this.cloudGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(4200), 3));
    this.cloudGeometry.setAttribute('color', new THREE.BufferAttribute(new Float32Array(4200), 3));
    this.cloud = new THREE.Points(this.cloudGeometry, new THREE.PointsMaterial({size:.006,vertexColors:true,transparent:true,opacity:.6,depthWrite:false,sizeAttenuation:true}));
    this.cloud.frustumCulled = false; this.cloudGeometry.setDrawRange(0, 0); this.group.add(this.cloud);
    this.frustum = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({color:0x679aab,transparent:true,opacity:.24,depthWrite:false}));
    this.group.add(this.frustum);
    this.targets = new Map();
    this.path = new THREE.Line(new THREE.BufferGeometry(),new THREE.LineBasicMaterial({color:0x68f1e3,transparent:true,opacity:.85,depthWrite:false}));
    this.group.add(this.path);
    this.waypoint = new THREE.Mesh(new THREE.OctahedronGeometry(.008),new THREE.MeshBasicMaterial({color:0xa7fff4,wireframe:true}));
    this.group.add(this.waypoint);
    this.shock = new THREE.Mesh(new THREE.RingGeometry(.94,1,80),new THREE.MeshBasicMaterial({color:0xff365b,transparent:true,opacity:0,side:THREE.DoubleSide,depthWrite:false,blending:THREE.AdditiveBlending}));
    scene.add(this.shock); this.shockAt = -1000; this.shattered = false;
  }

  target(name, size) {
    const group = new THREE.Group(), vertices = [];
    const half = size.map(v=>v/2);
    // Three short edges at each corner preserve visible object surfaces.
    for (const x of [-1,1]) for (const y of [-1,1]) for (const z of [-1,1]) {
      const p=[x*half[0],y*half[1],z*half[2]];
      for(let axis=0;axis<3;axis++) { const q=[...p];q[axis]*=.48;vertices.push(...p,...q); }
    }
    const geometry = new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(vertices,3));
    group.add(new THREE.LineSegments(geometry,new THREE.LineBasicMaterial({color:colors[name]||0x93d5b4,transparent:true,opacity:.9})));
    const axes = new THREE.AxesHelper(Math.max(...size)*.7);group.add(axes);
    this.group.add(group);this.targets.set(name,group);return group;
  }

  update(frame) {
    const sensor=frame.sensor;
    if(sensor?.provenance==='actual_corrupted_rgbd') {
      const count=Math.min(1400,Math.floor(sensor.points.length/3));
      this.cloudGeometry.attributes.position.array.set(sensor.points.slice(0,count*3));
      this.cloudGeometry.attributes.color.array.set(sensor.colors.slice(0,count*3));
      this.cloudGeometry.attributes.position.needsUpdate=true;this.cloudGeometry.attributes.color.needsUpdate=true;
      this.cloudGeometry.setDrawRange(0,count);
      const corners=sensor.frustum_corners, position=sensor.camera_pos, lines=[];
      for(let i=0;i<4;i++)lines.push(...position,...corners[i],...corners[i],...corners[(i+1)%4]);
      this.frustum.geometry.dispose();this.frustum.geometry=new THREE.BufferGeometry();
      this.frustum.geometry.setAttribute('position',new THREE.Float32BufferAttribute(lines,3));
    }
    for(const target of this.targets.values())target.visible=false;
    for(const [name,estimate] of Object.entries(frame.estimates||{})) {
      if(!valid(estimate.pos))continue;
      const size=estimate.size || (name==='core'?[.05,.05,.05]:name==='rubble_a'?[.095,.073,.045]:[.125,.062,.055]);
      const target=this.targets.get(name)||this.target(name,size);
      target.position.fromArray(estimate.pos);target.rotation.z=Number.isFinite(estimate.yaw)?estimate.yaw:0;target.visible=true;
      target.children[1].visible=estimate.orientation_observed!==false;
    }
    const origin=frame.fleet?.collector_tcp_m || frame.ee, goal=frame.planned_target;
    this.path.visible=this.waypoint.visible=valid(origin)&&valid(goal);
    if(this.path.visible) {
      const a=new THREE.Vector3(...origin),b=new THREE.Vector3(...goal);
      // Collinear interpolation draws only the issued segment, never a fictional
      // future lift/dump arc that the controller has not planned.
      const curve=new THREE.CatmullRomCurve3([a,a.clone().lerp(b,.33),a.clone().lerp(b,.67),b]);
      this.path.geometry.dispose();this.path.geometry=new THREE.BufferGeometry().setFromPoints(curve.getPoints(24));
      this.waypoint.position.copy(b);
    }
    if(frame.core_shattered&&!this.shattered) {
      this.shockAt=performance.now();
      this.shock.position.fromArray(frame.objects?.core?.pos||frame.ee||[.42,0,.42]);
      this.shock.position.z+=.003;
    }
    this.shattered=Boolean(frame.core_shattered);
    if(!this.shattered)this.shock.material.opacity=0;
  }

  tick(now) {
    const age=(now-this.shockAt)/1000;
    if(this.shattered&&age>=0&&age<1.3) {
      this.shock.scale.setScalar(.025+age*.23);this.shock.material.opacity=.65*(1-age/1.3);
    }else this.shock.material.opacity=0;
  }

  reset() {
    this.cloudGeometry.setDrawRange(0,0);this.frustum.geometry.dispose();this.frustum.geometry=new THREE.BufferGeometry();
    for(const target of this.targets.values())target.visible=false;
    this.path.visible=this.waypoint.visible=false;this.shattered=false;this.shock.material.opacity=0;
  }
}
