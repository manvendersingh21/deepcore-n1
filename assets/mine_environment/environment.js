import * as THREE from 'three';

// Synthetic presentation context; never a sensor or dynamics input.
export async function loadMineEnvironment(parent) {
  const group = new THREE.Group(); group.name = 'synthetic-mine-presentation';
  const geometries = [], materials = [], textures = [];
  let disposed = false;
  const dispose = () => {
    if (disposed) return;
    disposed = true; group.removeFromParent();
    for (const geometry of geometries) geometry.dispose();
    for (const material of materials) material.dispose();
    for (const texture of textures) texture.dispose();
  };
  try {
    const response = await fetch('/assets/mine_environment/environment.json');
    if (!response.ok) throw new Error(`Environment asset returned ${response.status}`);
    const metadata = await response.json();
    const rock = new THREE.MeshStandardMaterial({color:0x55595b,roughness:1,metalness:0,flatShading:true,side:THREE.DoubleSide});
    const steel = new THREE.MeshStandardMaterial({color:0x454c4d,roughness:.65,metalness:.7});
    const pipe = new THREE.MeshStandardMaterial({color:0x514a3f,roughness:.72,metalness:.5});
    const cable = new THREE.MeshStandardMaterial({color:0x12191c,roughness:.9});
    const lamp = new THREE.MeshStandardMaterial({color:0xffe1ae,emissive:0xffd49b,emissiveIntensity:3,roughness:.45});
    materials.push(rock,steel,pipe,cable,lamp);
    const byName={rock,steel,pipe,cable,lamp};
    const loader = new THREE.TextureLoader();
    for (const [slot,file] of [['map','Color'],['normalMap','NormalGL']]) {
      const texture = loader.load(`/assets/rock_ambientcg_Rock064/Rock064_1K-JPG_${file}.jpg`,
        value=>{if(disposed)value.dispose();},undefined,()=>{rock[slot]=null;rock.needsUpdate=true;});
      textures.push(texture);texture.wrapS=texture.wrapT=THREE.RepeatWrapping;
      if(slot==='map')texture.colorSpace=THREE.SRGBColorSpace;
      rock[slot]=texture;
    }
    rock.normalScale=new THREE.Vector2(.45,.45);
    for (const item of metadata.meshes) {
      if(item.name.endsWith('_unit'))continue; // templates for actual named robot-scene objects only
      const geometry=new THREE.BufferGeometry();geometries.push(geometry);
      geometry.setAttribute('position',new THREE.Float32BufferAttribute(item.vertices.flat(),3));
      if(item.uv)geometry.setAttribute('uv',new THREE.Float32BufferAttribute(item.uv.flat(),2));
      geometry.setIndex(item.indices);geometry.computeVertexNormals();geometry.computeBoundingSphere();
      const mesh=new THREE.Mesh(geometry,byName[item.material]);mesh.name=`presentation_mine_${item.name}`;
      mesh.castShadow=mesh.receiveShadow=true;group.add(mesh);
    }
    // Static supports/services batch by shape/material to keep draw calls bounded.
    const batches=new Map();
    for(const item of metadata.parts){
      const key=`${item.fromto?'beam':'box'}:${item.material}`;
      if(!batches.has(key))batches.set(key,[]);
      batches.get(key).push(item);
    }
    const transform=new THREE.Object3D();
    for(const [key,items] of batches){
      const beam=key.startsWith('beam:');
      const geometry=beam?new THREE.CylinderGeometry(1,1,1,8):new THREE.BoxGeometry(1,1,1);
      geometries.push(geometry);
      const instances=new THREE.InstancedMesh(geometry,byName[items[0].material],items.length);
      for(let index=0;index<items.length;index++){
        const item=items[index];transform.quaternion.identity();
        if(beam){
          const start=new THREE.Vector3(...item.fromto.slice(0,3)),end=new THREE.Vector3(...item.fromto.slice(3));
          const vector=end.clone().sub(start);
          transform.position.copy(start).add(end).multiplyScalar(.5);
          transform.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),vector.clone().normalize());
          transform.scale.set(item.radius,vector.length(),item.radius);
        }else{transform.position.fromArray(item.pos);transform.scale.fromArray(item.size.map(v=>v*2));}
        transform.updateMatrix();instances.setMatrixAt(index,transform.matrix);
      }
      instances.instanceMatrix.needsUpdate=true;instances.computeBoundingSphere();
      instances.name=`presentation_mine_batch_${key}`;instances.castShadow=instances.receiveShadow=true;group.add(instances);
    }
    for(const item of metadata.lights){
      const light=new THREE.SpotLight(new THREE.Color(...item.color),item.intensity,item.distance,.85,.8,2);
      light.position.fromArray(item.pos);light.target.position.fromArray(item.target);light.castShadow=false;
      group.add(light,light.target);
    }
    group.userData.provenance=metadata.provenance;parent.add(group);
    return {group,metadata,dispose};
  }catch(error){dispose();console.warn('Mine presentation environment unavailable',error);return {group,metadata:null,error,dispose};}
}
