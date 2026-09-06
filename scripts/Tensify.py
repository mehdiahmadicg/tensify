import maya.cmds as cmds
import maya.api.OpenMaya as om

# Create the Tensify node and attributes 
sel = cmds.ls(sl=True)
if not sel:
    cmds.error("Please select a mesh before running the script.")

obj = sel[0]

attributes = {
    "stretch": "stretchMap",
    "compression": "compressionMap"
}

tensify = cmds.createNode("tensify")

# Connect Tensify to the selected mesh
shape = cmds.listRelatives(obj, s=True, f=True)[0]
input_plug = cmds.listConnections(shape + ".inMesh", s=True, d=False, plugs=True)[0]  
source, out_attr = input_plug.split(".", 1)  

ref = cmds.duplicate(obj, name=obj + "_tensifyRef")[0]
ref_shape = cmds.listRelatives(ref, s=True, f=True)[0]
cmds.connectAttr(ref_shape + ".worldMesh[0]", tensify + ".refInput")
cmds.setAttr(ref + ".visibility", l=False)
cmds.setAttr(ref + ".visibility", 0)

cmds.connectAttr(f"{source}.{out_attr}", f"{tensify}.input")
cmds.connectAttr(f"{tensify}.output", f"{shape}.inMesh", f=True)

cmds.setAttr(ref_shape + ".caching", 1)

cmds.connectAttr(tensify + ".tensifyShader", shape + ".displayColors", f=True)
cmds.setDrivenKeyframe(ref_shape + ".nodeState", cd=tensify + ".tensifyShader", dv=1, v=0)
cmds.setDrivenKeyframe(ref_shape + ".nodeState", cd=tensify + ".tensifyShader", dv=0, v=1)

# Connect 'stretchMap' and 'compressionMap' attributes
for primvar, tensify_attr in attributes.items():

    main_attr = f"{shape}.{primvar}"
    geom_scope_attr = f"{main_attr}_AbcGeomScope"
    abc_type_attr = f"{main_attr}_AbcType"

    if not cmds.objExists(main_attr):
        cmds.addAttr(shape, longName=primvar, dataType="doubleArray")

    if not cmds.objExists(geom_scope_attr):
        cmds.addAttr(shape, longName=f"{primvar}_AbcGeomScope", dataType="string")
        cmds.setAttr(geom_scope_attr, "var", type="string")

    if not cmds.objExists(abc_type_attr):
        cmds.addAttr(shape, longName=f"{primvar}_AbcType", dataType="string")
        cmds.setAttr(abc_type_attr, "", type="string")

    src_attr = f"{tensify}.{tensify_attr}"
    if cmds.objExists(src_attr):
        cmds.connectAttr(src_attr, main_attr, force=True)
    else:
        print(f"Source attribute {src_attr} not found.")
                
# Set the 'smoothingRadius' limitation
sel = om.MSelectionList()
sel.add(tensify + ".smoothingRadius")
plug = sel.getPlug(0)
attr = plug.attribute()

fn = om.MFnNumericAttribute(attr)
fn.setMin(0)
fn.setMax(1000)
     
