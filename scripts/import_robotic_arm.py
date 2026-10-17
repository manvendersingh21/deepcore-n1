"""Inspect/export user-provided rigid GLB parts without inventing joint rigging."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def convert(source=ROOT / 'dev/robotic_arm.glb', output=ROOT / 'assets/robot_reference/user_arm'):
    source, output = Path(source), Path(output)
    raw = source.read_bytes()
    magic, version, length = struct.unpack_from('<4sII', raw)
    if magic != b'glTF' or version != 2 or length != len(raw):
        raise ValueError('Expected valid GLB2')
    offset, document, binary = 12, None, None
    while offset < len(raw):
        size, kind = struct.unpack_from('<I4s', raw, offset)
        data = raw[offset + 8:offset + 8 + size]
        if kind == b'JSON': document = json.loads(data)
        if kind == b'BIN\x00': binary = data
        offset += 8 + size
    if document is None or binary is None or document.get('extensionsRequired'):
        raise ValueError('Unsupported GLB buffer/required extension')
    output.mkdir(parents=True, exist_ok=True)
    def accessor(index):
        a = document['accessors'][index]
        if 'sparse' in a: raise ValueError('Sparse accessors require explicit conversion')
        view = document['bufferViews'][a['bufferView']]
        dtype = {5120:'i1',5121:'u1',5122:'<i2',5123:'<u2',5125:'<u4',5126:'<f4'}[a['componentType']]
        columns = {'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4}[a['type']]
        item = np.dtype(dtype).itemsize
        return np.ndarray((a['count'],columns), dtype=dtype, buffer=binary,
                          offset=view.get('byteOffset',0)+a.get('byteOffset',0),
                          strides=(view.get('byteStride',columns*item),item)).copy()
    matrices = {}
    def matrix(node):
        if 'matrix' in node: return np.array(node['matrix']).reshape(4,4).T
        t = np.eye(4);t[:3,3] = node.get('translation',[0,0,0])
        x,y,z,w = node.get('rotation',[0,0,0,1])
        r = np.eye(4);r[:3,:3] = [[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]
        scale = np.diag([*node.get('scale',[1,1,1]),1])
        return t@r@scale
    def visit(index,parent):
        matrices[index] = parent@matrix(document['nodes'][index])
        for child in document['nodes'][index].get('children',[]): visit(child,matrices[index])
    for index in document['scenes'][document.get('scene',0)]['nodes']:visit(index,np.eye(4))
    mtl=[];images=[]
    for index,material in enumerate(document.get('materials',[])):
        pbr=material.get('pbrMetallicRoughness',{})
        color=pbr.get('baseColorFactor',[1,1,1,1])
        lines=[f'newmtl material_{index}', 'Kd '+' '.join(map(str,color[:3])),f'd {color[3]}',
               f'Pr {pbr.get("roughnessFactor",1)}',f'Pm {pbr.get("metallicFactor",1)}']
        if 'baseColorTexture' in pbr:
            image_index=document['textures'][pbr['baseColorTexture']['index']]['source']
            image=document['images'][image_index];view=document['bufferViews'][image['bufferView']]
            data=binary[view.get('byteOffset',0):view.get('byteOffset',0)+view['byteLength']]
            ext='png' if image['mimeType']=='image/png' else 'jpg'
            filename=f'basecolor_{index}.{ext}';(output/filename).write_bytes(data)
            lines.append('map_Kd '+filename)
            images.append({'material':index,'source_image':image_index,'file':filename,'sha256':hashlib.sha256(data).hexdigest()})
        mtl.append('\n'.join(lines))
    (output/'materials.mtl').write_text('\n\n'.join(mtl)+'\n')
    parts=[];preview=[]
    for node_index,transform in matrices.items():
        node=document['nodes'][node_index]
        if 'mesh' not in node:continue
        for primitive_index,primitive in enumerate(document['meshes'][node['mesh']]['primitives']):
            if primitive.get('mode',4)!=4:raise ValueError('Only explicit triangle meshes supported')
            local=accessor(primitive['attributes']['POSITION'])
            positions=np.column_stack([local,np.ones(len(local))])@transform.T
            positions=positions[:,:3]
            faces=(accessor(primitive['indices']).reshape(-1,3) if 'indices' in primitive else np.arange(len(local)).reshape(-1,3))
            if np.linalg.det(transform[:3,:3])<0:faces=faces[:,::-1]
            attributes=primitive['attributes']
            uv=accessor(attributes['TEXCOORD_0']) if 'TEXCOORD_0' in attributes else None
            normals=accessor(attributes['NORMAL']) if 'NORMAL' in attributes else None
            if normals is not None:
                normals=normals@np.linalg.inv(transform[:3,:3]);normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
            name=f'part_{node_index:03d}_{primitive_index}'
            lines=['mtllib materials.mtl',f'o {name}',f'usemtl material_{primitive.get("material",0)}']
            lines+=['v %.8g %.8g %.8g'%tuple(v) for v in positions]
            if uv is not None:lines+=['vt %.8g %.8g'%tuple(v) for v in uv]
            if normals is not None:lines+=['vn %.8g %.8g %.8g'%tuple(v) for v in normals]
            def face_vertex(v):
                i=int(v)+1
                return f'{i}/{i}/{i}' if uv is not None and normals is not None else f'{i}/{i}' if uv is not None else f'{i}//{i}' if normals is not None else str(i)
            lines+=['f '+' '.join(face_vertex(v) for v in face) for face in faces]
            (output/f'{name}.obj').write_text('\n'.join(lines)+'\n')
            parts.append({'file':f'{name}.obj','source_node':node_index,'source_name':node.get('name'),
                          'source_mesh':node['mesh'],'primitive':primitive_index,'material':primitive.get('material'),
                          'authored_world_matrix':transform.tolist(),'bounds':[positions.min(0).tolist(),positions.max(0).tolist()],
                          'vertices':len(positions),'triangles':len(faces)})
            preview.append((positions,faces,primitive.get('material',0)))
    bounds=np.array([part['bounds'] for part in parts])
    metadata={'source_file':str(source),'source_sha256':hashlib.sha256(raw).hexdigest(),'asset':document['asset'],
              'nodes':document['nodes'],'materials':document.get('materials',[]),'extracted_basecolors':images,
              'skins':len(document.get('skins',[])),'animations':len(document.get('animations',[])),
              'parts':parts,'world_bounds':[bounds[:,0].min(0).tolist(),bounds[:,1].max(0).tolist()],
              'coordinate_policy':'Authored glTF world positions; node transforms baked once. No scale normalization or invented articulation.',
              'rig_assessment':'Static generic part hierarchy; no six-joint link/pivot/axis specification. Not safely mappable to UR5e automatically.'}
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    # CPU-only orthographic inspection of actual triangles, not an animation.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    figure,axes=plt.subplots(1,3,figsize=(15,5),facecolor='#15191c')
    palette=['#8a785e','#b48150','#778287','#a4aaad','#292e32']
    for axis,(horizontal,vertical,label) in zip(axes,[(0,1,'XY'),(0,2,'XZ'),(2,1,'ZY')]):
        axis.set_facecolor('#20262a')
        for vertices,faces,material in preview:
            axis.add_collection(PolyCollection(vertices[faces][:,:,[horizontal,vertical]],facecolors=palette[material%5],edgecolors='none',alpha=.8))
        axis.autoscale();axis.set_aspect('equal');axis.set_title(label,color='white');axis.tick_params(colors='#b5c1c7')
    figure.suptitle('User GLB: actual static triangle geometry / no authored joint rig',color='white')
    figure.tight_layout();figure.savefig(output/'inspection.png',dpi=130);plt.close(figure)
    return metadata


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,default=ROOT/'dev/robotic_arm.glb');parser.add_argument('--out',type=Path,default=ROOT/'assets/robot_reference/user_arm')
    args=parser.parse_args();result=convert(args.source,args.out)
    print(json.dumps({'parts':len(result['parts']),'bounds':result['world_bounds'],'skins':result['skins'],'animations':result['animations'],'source_sha256':result['source_sha256']},indent=2))
