import * as THREE from 'three';

// The downloaded survey is presentation context. It never supplies physics poses.
export async function loadCMUMine(parent) {
  const group = new THREE.Group();
  group.name = 'cmu-survey-cutaway';
  const geometries = [];
  const textures = [];
  const material = new THREE.MeshStandardMaterial({
    color: 0xb5aa99, roughness: 0.98, metalness: 0,
    normalScale: new THREE.Vector2(0.45, 0.45),
    side: THREE.DoubleSide, flatShading: false,
  });
  let disposed = false;
  const dispose = () => {
    if (disposed) return;
    disposed = true;
    group.removeFromParent();
    for (const geometry of geometries) geometry.dispose();
    for (const texture of textures) texture.dispose();
    material.dispose();
  };
  try {
    const base = '/assets/cmu_mine_reference/derived/';
    const responses = await Promise.all([fetch(`${base}tunnel.json`), fetch(`${base}metadata.json`)]);
    if (responses.some(response => !response.ok)) throw new Error('CMU derived survey assets unavailable');
    const [data, metadata] = await Promise.all(responses.map(response => response.json()));
    const loader = new THREE.TextureLoader();
    for (const [slot, file] of [['map', 'Color'], ['normalMap', 'NormalGL']]) {
      const texture = loader.load(`/assets/rock_ambientcg_Rock064/Rock064_1K-JPG_${file}.jpg`,
        loaded => { if (disposed) loaded.dispose(); }, undefined,
        error => {
          if (!disposed) {
            material[slot] = null;
            material.needsUpdate = true;
            console.warn(`CMU rock ${slot} unavailable; retaining survey surface.`, error);
          }
          texture.dispose();
        });
      texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
      if (slot === 'map') texture.colorSpace = THREE.SRGBColorSpace;
      textures.push(texture);
      material[slot] = texture;
    }
    for (const chunk of data.chunks) {
      const indexed = new THREE.BufferGeometry();
      indexed.setAttribute('position', new THREE.Float32BufferAttribute(chunk.vertices.flat(), 3));
      indexed.setIndex(chunk.indices);
      indexed.computeVertexNormals();
      // Separate UV seams per triangle while keeping all surveyed vertex positions/normals.
      const geometry = indexed.toNonIndexed();
      indexed.dispose();
      geometries.push(geometry);
      const positions = geometry.getAttribute('position');
      const uv = new Float32Array(positions.count * 2);
      const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3();
      const normal = new THREE.Vector3(), edge = new THREE.Vector3();
      for (let start = 0; start < positions.count; start += 3) {
        a.fromBufferAttribute(positions, start);
        b.fromBufferAttribute(positions, start + 1);
        c.fromBufferAttribute(positions, start + 2);
        normal.crossVectors(edge.subVectors(b, a), c.sub(a));
        const nx = Math.abs(normal.x), ny = Math.abs(normal.y), nz = Math.abs(normal.z);
        for (let i = start; i < start + 3; i++) {
          const x = positions.getX(i), y = positions.getY(i), z = positions.getZ(i);
          // World projection, 1.5 repeats per unit: XY floor, XZ/YZ walls.
          uv[i * 2] = (nz >= nx && nz >= ny ? x : nx >= ny ? y : x) * 1.5;
          uv[i * 2 + 1] = (nz >= nx && nz >= ny ? y : z) * 1.5;
        }
      }
      geometry.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
      geometry.computeBoundingSphere();
      const mesh = new THREE.Mesh(geometry, material);
      mesh.name = `cmu_${chunk.name}`;
      mesh.castShadow = mesh.receiveShadow = true;
      group.add(mesh);
    }
    group.userData.source = metadata;
    parent.add(group);
    return { group, metadata, dispose };
  } catch (error) {
    dispose();
    console.warn('CMU survey context unavailable; using simulation scene.', error);
    return { group, metadata: null, error, dispose };
  }
}
