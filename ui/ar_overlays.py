"""Native presentation model isolation and overlays from actual sensor/policy data."""
from __future__ import annotations
import os
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _industrial_detail(xml):
    """Noncolliding casing details on the independent presentation model only."""
    root = ET.fromstring(xml)
    assets = root.find('asset')
    for name, rgba in [('presentation_rubber', '.035 .043 .05 1'),
                       ('presentation_fastener', '.48 .55 .61 1')]:
        ET.SubElement(assets, 'material', name=name, rgba=rgba,
                      specular='.42' if name.endswith('fastener') else '.04', shininess='.3')
    for name in ('drill_mobile_base', 'collector_mobile_base'):
        body = root.find(f".//body[@name='{name}']")
        if body is None: continue
        geom = body.find(f"geom[@name='{name}']")
        x,y,z = .18,.12,.045
        bevel=.016
        outline=[(-x+bevel,-y),(x-bevel,-y),(x,-y+bevel),(x,y-bevel),
                 (x-bevel,y),(-x+bevel,y),(-x,y-bevel),(-x,-y+bevel)]
        vertices=[(a,b,c) for c in (-z,z) for a,b in outline]
        faces=[]
        for i in range(1,7): faces.extend([(0,i+1,i),(8,8+i,9+i)])
        for i in range(8):
            j=(i+1)%8;faces.extend([(i,j,8+j),(i,8+j,8+i)])
        meshname=f'presentation_{name}_casing'
        ET.SubElement(assets,'mesh',name=meshname,vertex=' '.join(str(v) for p in vertices for v in p),
                      face=' '.join(str(v) for p in faces for v in p))
        geom.set('type','mesh');geom.set('mesh',meshname);geom.attrib.pop('size',None)
        for side in (-1,1):
            for i in range(6):
                ET.SubElement(body,'geom',type='box',name=f'presentation_{name}_vent_{side}_{i}',
                              pos=f'{-.12+i*.025} {side*.121} .012',size='.007 .002 .018',
                              material='presentation_rubber',contype='0',conaffinity='0',mass='0',group='2')
            for i in range(5):
                ET.SubElement(body,'geom',type='cylinder',name=f'presentation_{name}_hub_{side}_{i}',
                              pos=f'{-.14+i*.07} {side*.183} -.025',size='.009 .003',
                              quat='.70710678 .70710678 0 0',material='presentation_fastener',
                              contype='0',conaffinity='0',mass='0',group='2')
            for x in (-.14,.14):
                ET.SubElement(body,'geom',type='cylinder',pos=f'{x} {side*.09} .046',size='.005 .002',
                              material='presentation_fastener',contype='0',conaffinity='0',mass='0',group='2')
    return ET.tostring(root, encoding='unicode')


class VisualModel:
    """Private decorated model/MjData; never mutates the robot observation model."""
    def __init__(self, source_model):
        from ui.mine_environment import append_presentation_environment
        fd, filename = tempfile.mkstemp(prefix='.native_presentation_', suffix='.xml', dir=ROOT)
        os.close(fd)
        try:
            mujoco.mj_saveLastXML(filename, source_model)
            xml = append_presentation_environment(_industrial_detail(Path(filename).read_text()))
            self.model = mujoco.MjModel.from_xml_string(xml)
        finally:
            Path(filename).unlink(missing_ok=True)
        self.data = mujoco.MjData(self.model)
        if self.model.nq != source_model.nq or self.model.nv != source_model.nv or self.model.nu != source_model.nu:
            raise RuntimeError('Presentation changed the physical state layout')
        pairs = []
        for source in range(source_model.ngeom):
            name = mujoco.mj_id2name(source_model, mujoco.mjtObj.mjOBJ_GEOM, source)
            if name:
                target = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
                if target >= 0:pairs.append((source,target))
        self.alpha_pairs = np.asarray(pairs,dtype=int).reshape(-1,2)

    def sync(self, ep):
        source=ep.data;target=self.data
        for name in ('qpos','qvel','act','ctrl','mocap_pos','mocap_quat','eq_active','userdata'):
            a,b=getattr(source,name,None),getattr(target,name,None)
            if a is not None and b is not None and a.shape==b.shape:b[:]=a
        target.time=source.time
        pairs=self.alpha_pairs
        self.model.geom_rgba[pairs[:,1],3]=ep.model.geom_rgba[pairs[:,0],3]
        mujoco.mj_forward(self.model,target)
        return target


def _line(scene,a,b,color,width=1.2):
    if scene.ngeom>=scene.maxgeom:return
    a,b=np.asarray(a,float),np.asarray(b,float)
    if not np.isfinite(a).all() or not np.isfinite(b).all() or np.linalg.norm(a-b)<1e-8:return
    geom=scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_LINE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.asarray(color,np.float32))
    mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_LINE,width,a,b)
    geom.category=mujoco.mjtCatBit.mjCAT_DECOR;geom.objtype=mujoco.mjtObj.mjOBJ_UNKNOWN;geom.objid=-1
    geom.emission=.4;scene.ngeom+=1


def add_headlamp(scene, ep):
    """Light origin follows the actual articulated wrist, with a visible beam outline."""
    site=-1
    for name in ('collector_grip_site','grip_site','tool_tip','wrist_site'):
        site=mujoco.mj_name2id(ep.model,mujoco.mjtObj.mjOBJ_SITE,name)
        if site>=0:break
    if site<0:return
    p=np.asarray(ep.data.site_xpos[site],float)
    rotation=np.asarray(ep.data.site_xmat[site]).reshape(3,3)
    direction=-rotation[:,2]
    # Keep the lamp on the tool frame; no target-object ground truth is used.
    if direction[2]>.5:direction=-direction
    origin=p+rotation[:,0]*.035-direction*.025
    if scene.nlight<len(scene.lights):
        light=scene.lights[scene.nlight]
        light.pos[:]=origin;light.dir[:]=direction
        if hasattr(light,'type'):light.type=mujoco.mjtLightType.mjLIGHT_SPOT
        else:light.directional=False
        if hasattr(light,'intensity'):light.intensity=1
        light.headlight=False
        light.diffuse[:]=(.85,.93,1);light.specular[:]=(.25,.3,.35);light.ambient[:]=0
        light.cutoff=24;light.exponent=12;light.attenuation[:]=(1,0,4);light.castshadow=True
        scene.nlight+=1
    right=rotation[:,0];up=np.cross(direction,right)
    end=origin+direction*.48
    ring=[]
    for angle in np.linspace(0,2*np.pi,13):
        ring.append(end+.105*(right*np.cos(angle)+up*np.sin(angle)))
    for a,b in zip(ring,ring[1:]):_line(scene,a,b,(.65,.83,1,.16),.6)
    for point in ring[::3]:_line(scene,origin,point,(.65,.83,1,.13),.6)


def add_sensor_overlays(scene, packet=None, estimates=None, command=None, ee=None, core_force=None):
    """Observed point cloud/frustum, policy estimates and issued command only."""
    if packet and packet.get('provenance')=='actual_corrupted_rgbd':
        points=np.asarray(packet.get('points',[]),float).reshape(-1,3)
        colors=np.asarray(packet.get('colors',[]),float).reshape(-1,3)
        stride=max(1,int(np.ceil(len(points)/380)))
        for point,color in zip(points[::stride],colors[::stride]):
            if scene.ngeom>=scene.maxgeom:break
            geom=scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.002,.002,.002]),point,np.eye(3).ravel(),np.r_[color,.65].astype(np.float32))
            geom.category=mujoco.mjtCatBit.mjCAT_DECOR;geom.objtype=mujoco.mjtObj.mjOBJ_UNKNOWN;geom.objid=-1;geom.emission=.25;scene.ngeom+=1
        origin=packet.get('camera_pos');corners=packet.get('frustum_corners',[])
        if origin is not None and len(corners)==4:
            for corner in corners:_line(scene,origin,corner,(.3,.75,.85,.18),.7)
            for i in range(4):_line(scene,corners[i],corners[(i+1)%4],(.3,.75,.85,.25),.8)
    for name,value in (estimates or {}).items():
        pos=np.asarray(value.get('pos',[]),float)
        if pos.shape!=(3,) or not np.isfinite(pos).all():continue
        yaw=float(value.get('yaw',0));c,s=np.cos(yaw),np.sin(yaw)
        axes=[np.array([c,s,0]),np.array([-s,c,0]),np.array([0,0,1])]
        if value.get('orientation_observed', True):
            for axis,color in zip(axes,[(.95,.3,.3,.9),(.3,.9,.4,.9),(.25,.65,1,.9)]):_line(scene,pos,pos+axis*.045,color,1.4)
        # Nominal object-size estimate bracket, never ground-truth extents.
        half=.027 if name=='core' else .038
        corners=[pos+axes[0]*x*half+axes[1]*y*half+axes[2]*z*.025 for z in [-1,1] for y in [-1,1] for x in [-1,1]]
        for a,b in [(0,1),(0,2),(1,3),(2,3),(4,5),(4,6),(5,7),(6,7),(0,4),(1,5),(2,6),(3,7)]:_line(scene,corners[a],corners[b],(.2,.9,.95,.65),1.1)
        if name=='core' and core_force is not None:
            force=float(core_force);color=(1,.2,.12,.9) if force>18 else (.2,.9,.95,.7)
            ring=[pos+np.array([.043*np.cos(a),.043*np.sin(a),.036]) for a in np.linspace(0,2*np.pi,33)]
            for a,b in zip(ring,ring[1:]):_line(scene,a,b,color,1.8)
    if command is not None and ee is not None:
        target=np.asarray(command,float)
        if target.shape==(3,):_line(scene,ee,target,(.96,.69,.25,.9),2)


def add_force_arrow(scene,point,force,limit):
    force=np.asarray(force,float);point=np.asarray(point,float)
    if force.shape!=(3,) or point.shape!=(3,) or not np.isfinite(force).all():return
    magnitude=float(np.linalg.norm(force))
    if magnitude<.5 or scene.ngeom>=scene.maxgeom:return
    color=(1,.2,.12,.95) if magnitude>limit else ((1,.65,.15,.95) if magnitude>=min(10.,limit*.7) else (.2,.95,.8,.95))
    geom=scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_ARROW,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.asarray(color,np.float32))
    mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_ARROW,.004,point,point+force/magnitude*min(.23,.04+magnitude*.004))
    geom.category=mujoco.mjtCatBit.mjCAT_DECOR;geom.objtype=mujoco.mjtObj.mjOBJ_UNKNOWN;geom.objid=-1;geom.emission=.4
    scene.ngeom+=1
