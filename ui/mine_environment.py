"""Synthetic geological surround, applied ONLY to presentation copies.

The CMU survey remains a separate source-derived mesh. These deterministic rock
surfaces/supports/cables give the workcell enclosing context, not measured geology.
Never call this on the physical model used to generate robot observations.
"""
from __future__ import annotations
import json
import math
import re
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'assets/mine_environment'


def generate_assets():
    ASSETS.mkdir(parents=True, exist_ok=True)
    # MuJoCo's XML image loader supports PNG; derive from the existing CC0 color map.
    from PIL import Image
    Image.open(ROOT/'assets/rock_ambientcg_Rock064/Rock064_1K-JPG_Color.jpg').save(ASSETS/'rock_color.png')
    meshes = []
    def mesh(name, vertices, faces, material, uv=None):
        v, f = np.asarray(vertices), np.asarray(faces)
        # Two independent sides preserve native visibility without wall thickness.
        obj = ''.join('v %.6f %.6f %.6f\n' % tuple(point) for point in np.vstack([v,v]))
        if uv is not None:
            obj += ''.join('vt %.6f %.6f\n' % tuple(point) for point in np.vstack([uv,uv]))
        normals = np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]])
        normals /= np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
        obj += ''.join('vn %.6f %.6f %.6f\n' % tuple(n) for n in np.vstack([normals,-normals]))
        for side in (0,1):
            for face_index,face in enumerate(f):
                ids = (face if side == 0 else face[::-1]) + 1 + side*len(v)
                normal_index=face_index+1+side*len(f)
                obj += 'f ' + ' '.join(f'{i}/{i}/{normal_index}' if uv is not None else f'{i}//{normal_index}' for i in ids) + '\n'
        (ASSETS/f'{name}.obj').write_text(obj)
        meshes.append({'name':name,'vertices':v.round(6).tolist(),'indices':f.ravel().tolist(),
                       'uv':np.asarray(uv).round(6).tolist() if uv is not None else None,
                       'material':material,'obj':f'{name}.obj'})
    # Back half of a continuous arched tunnel, plus full vaults beyond the work opening.
    xs=np.linspace(-5.4,4.2,97)
    theta=np.linspace(0,math.pi,49)
    vertices=[];uv=[]
    for x in xs:
        for a in theta:
            noise=.045*math.sin(x*7.1+a*13)+.028*math.sin(x*17-a*21)+.022*math.cos(x*31+a*29)
            radius=2.12+noise
            vertices.append([x,radius*math.cos(a),.4+(2.28+noise)*math.sin(a)])
            uv.append([x*.65,a*1.4])
    faces=[]
    for i in range(len(xs)-1):
        for j in range(len(theta)-1):
            center_x=(xs[i]+xs[i+1])/2;angle=(theta[j]+theta[j+1])/2
            # Front cutaway opens only the operator's workline, retaining real tunnel depth.
            if -3.15<center_x<3.45 and angle>math.pi*.52 and angle<math.pi*.965:
                continue
            a=i*len(theta)+j;b=a+len(theta)
            faces.extend([[a,b,a+1],[a+1,b,b+1]])
    mesh('rough_tunnel_vault',vertices,faces,'rock',uv)
    # Low rough floor skirt outside the physical floor; no raised work obstacles.
    vertices=[];uv=[];faces=[]
    ys=np.linspace(-2.12,2.12,29)
    for x in xs:
        for y in ys:
            z=.392-.009*(1+math.sin(13*x+9*y))
            vertices.append([x,y,z]);uv.append([x*.8,y*.8])
    for i in range(len(xs)-1):
        for j in range(len(ys)-1):
            x=(xs[i]+xs[i+1])/2;y=(ys[j]+ys[j+1])/2
            if -.95<x<2.85 and abs(y)<1.1:continue
            a=i*len(ys)+j;b=a+len(ys)
            faces.extend([[a,a+1,b],[a+1,b+1,b]])
    mesh('rough_floor_skirt',vertices,faces,'rock',uv)
    # Unit-envelope faceted replacement meshes; actual poses stay with source geoms.
    from scipy.spatial import ConvexHull
    rng=np.random.default_rng(7314)
    points=rng.normal(size=(38,3));points/=np.linalg.norm(points,axis=1,keepdims=True)
    points*=rng.uniform(.72,1,(38,1));points/=np.max(np.abs(points),axis=0)
    hull=ConvexHull(points)
    faces=hull.simplices.copy()
    for i,face in enumerate(faces):
        a,b,c=points[face]
        if np.dot(np.cross(b-a,c-a),(a+b+c)/3)<0:faces[i]=face[::-1]
    mesh('faceted_basalt_unit',points,faces,'rock')
    vertices=[];faces=[]
    for crystal,(cx,cy,height,radius) in enumerate([(0,0,1,.42),(.44,.16,.64,.25),(-.4,-.22,.72,.28)]):
        start=len(vertices)
        for z,shrink in [(-.9,1),(height*.5,.8)]:
            for i in range(6):
                a=i*math.pi/3+crystal*.3;vertices.append([cx+radius*shrink*math.cos(a),cy+radius*shrink*math.sin(a),z])
        vertices.append([cx+.08,cy-.05,height])
        for i in range(6):
            j=(i+1)%6
            faces.extend([[start+i,start+j,start+6+i],[start+j,start+6+j,start+6+i],
                          [start+6+i,start+6+j,start+12]])
        for i in range(1,5):faces.append([start,start+i+1,start+i])
    mesh('cyan_crystal_unit',vertices,faces,'crystal')
    parts=[]
    def part(name,kind,material,**values):parts.append({'name':name,'type':kind,'material':material,**values})
    def beam(name,a,b,r,material='steel'):
        part(name,'capsule',material,fromto=[*a,*b],radius=r)
    # Mine-scale arched steel sets, open on the viewer side around the work cell.
    for index,x in enumerate(np.arange(-5.1,4.15,1.15)):
        angles=np.linspace(0,math.pi,19)
        for j in range(len(angles)-1):
            a,b=angles[j:j+2]
            if -3.15<x<3.45 and a>math.pi*.52 and b<math.pi*.965:continue
            p=[x,2.035*math.cos(a),.4+2.19*math.sin(a)]
            q=[x,2.035*math.cos(b),.4+2.19*math.sin(b)]
            beam(f'support_{index}_{j}',p,q,.032)
        for side in [-1,1]:
            part(f'footing_{index}_{side}','box','steel',pos=[x,side*2.04,.425],size=[.10,.11,.025])
        for h in [1.1,1.8]:
            part(f'rock_bolt_{index}_{h}','box','steel',pos=[x,1.99,h],size=[.065,.012,.065])
    # Continuous service pipes and bundled cable following the far wall.
    for pipe,z in enumerate([.76,.89,1.94]):
        radius=.026 if pipe<2 else .012
        beam(f'pipe_{pipe}',[-5.25,1.91,z],[4.0,1.91,z],radius,'pipe' if pipe<2 else 'cable')
        for index,x in enumerate(np.arange(-5,4,.65)):
            part(f'pipe_clamp_{pipe}_{index}','box','steel',pos=[float(x),1.93,z],size=[.012,.045,.045])
    for cable in range(3):
        for index in range(35):
            a=-5.2+index*.26;b=a+.26
            def cablepoint(x):return [x,1.86+cable*.018,2.13-.035*math.sin((x+5.2)*math.pi/.78)]
            beam(f'cable_{cable}_{index}',cablepoint(a),cablepoint(b),.006,'cable')
    lights=[]
    for index,x in enumerate([-4.5,-2.4,-.2,2.1,3.8]):
        part(f'lamp_housing_{index}','box','steel',pos=[x,1.80,1.95],size=[.13,.055,.055])
        part(f'lamp_face_{index}','box','lamp',pos=[x,1.74,1.95],size=[.105,.005,.036])
        lights.append({'name':f'worklight_{index}','pos':[x,1.68,1.93],'color':[1,.83,.62],
                       'target':[x,-.3,.5],'intensity':7,'distance':4.0})
    metadata={'provenance':'Synthetic presentation-only geological surround and industrial services; not CMU measured geometry.',
              'scan_relationship':'Keep the actual CMU survey cutaway as a separate source-derived layer.',
              'sensor_policy':'Never inserted into observation/physics model; native presentation copy and browser visualization only.',
              'floor_z':.4,'tunnel_axis':'X','bounds':[[-5.4,-2.2,.37],[4.2,2.2,2.8]],
              'front_cutaway':'Operator side -Y opened above low wall for x(-3.15,3.45); back wall/half vault and full end portals retained.',
              'meshes':meshes,'parts':parts,'lights':lights}
    (ASSETS/'environment.json').write_text(json.dumps(metadata,separators=(',',':'))+'\n')
    return metadata


def append_presentation_environment(xml: str) -> str:
    """Add noncolliding fixed geoms to a separate presentation model only.

    No body/joint/equality additions. Caller must use its own model/MjData and
    copy measured robot state. Existing observation model must remain untouched.
    """
    spec=json.loads((ASSETS/'environment.json').read_text())
    root=ET.fromstring(xml);world=root.find('worldbody')
    if world is None:raise ValueError('Presentation model needs worldbody')
    if world.find("geom[@name='presentation_mine_rough_tunnel_vault']") is not None:return xml
    asset=root.find('asset')
    if asset is None:asset=ET.SubElement(root,'asset')
    rock={'rgba':'.32 .34 .35 1','specular':'.025','shininess':'.02'}
    if asset.find("texture[@name='tex_rock']") is not None:rock.update(texture='tex_rock',texrepeat='1 1')
    else:
        texture=ASSETS/'rock_color.png'
        if texture.exists():
            ET.SubElement(asset,'texture',name='presentation_mine_rock_texture',type='2d',file=str(texture.resolve()))
            rock.update(texture='presentation_mine_rock_texture',texrepeat='1 1')
    materials={'rock':rock,'steel':{'rgba':'.24 .26 .25 1','specular':'.25','shininess':'.25'},
               'pipe':{'rgba':'.32 .29 .23 1','specular':'.18'},'cable':{'rgba':'.055 .064 .067 1'},
               'lamp':{'rgba':'1 .88 .68 1','emission':'.8'}}
    materials['crystal']={'rgba':'.035 .72 .82 1','emission':'.55','specular':'.65','shininess':'.7'}
    for name,attributes in materials.items():ET.SubElement(asset,'material',name='presentation_mine_'+name,**attributes)
    for item in spec['meshes']:
        if item['name'].endswith('_unit'):continue
        name='presentation_mine_'+item['name']
        ET.SubElement(asset,'mesh',name=name,file=str((ASSETS/item['obj']).resolve()),maxhullvert='32')
        ET.SubElement(world,'geom',name=name,type='mesh',mesh=name,material='presentation_mine_'+item['material'],
                      contype='0',conaffinity='0',density='0',group='2')
    # Replace only the visual shape of actual named material geoms, retaining
    # name/body/pose/alpha. Unit envelopes never exceed the original box bounds.
    for geom in root.findall('.//geom'):
        name=geom.get('name','')
        if name not in ('core','rubble_a','rubble_b','rock_base') and not re.fullmatch(r'debris_\d+',name):continue
        if geom.get('type')=='mesh':continue
        size=list(map(float,geom.get('size','').split()))
        if not size:continue
        kind=geom.get('type','box')
        if kind=='sphere':scale=[size[0]]*3
        elif kind in ('cylinder','capsule'):scale=[size[0],size[0],size[1]+(size[0] if kind=='capsule' else 0)]
        elif len(size)>=3:scale=size[:3]
        else:continue
        unit='cyan_crystal_unit' if name=='core' else 'faceted_basalt_unit'
        mesh_name='presentation_mine_shape_'+name
        ET.SubElement(asset,'mesh',name=mesh_name,file=str((ASSETS/(unit+'.obj')).resolve()),
                      scale=' '.join(map(str,scale)),maxhullvert='32')
        geom.set('type','mesh');geom.set('mesh',mesh_name);geom.attrib.pop('size',None)
        geom.set('material','presentation_mine_crystal' if name=='core' else 'presentation_mine_rock')
        geom.set('contype','0');geom.set('conaffinity','0');geom.set('group','2')
    for item in spec['parts']:
        values={'name':'presentation_mine_'+item['name'],'type':item['type'],
                'material':'presentation_mine_'+item['material'],'contype':'0','conaffinity':'0','density':'0','group':'2'}
        if 'fromto' in item:values.update(fromto=' '.join(map(str,item['fromto'])),size=str(item['radius']))
        else:values.update(pos=' '.join(map(str,item['pos'])),size=' '.join(map(str,item['size'])))
        ET.SubElement(world,'geom',**values)
    for light in spec['lights']:
        direction=np.array(light['target'])-light['pos'];direction/=np.linalg.norm(direction)
        ET.SubElement(world,'light',name='presentation_mine_'+light['name'],pos=' '.join(map(str,light['pos'])),
                      dir=' '.join(map(str,direction)),diffuse='.55 .46 .34',specular='.1 .09 .07',
                      cutoff='70',exponent='3',castshadow='false')
    return ET.tostring(root,encoding='unicode')


if __name__=='__main__':
    spec=generate_assets()
    print(json.dumps({'meshes':len(spec['meshes']),'parts':len(spec['parts']),'lights':len(spec['lights'])}))
