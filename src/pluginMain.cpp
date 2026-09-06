// pluginMain.cpp

#include "Tensify.h"

#include <maya/MFnPlugin.h>
#include <maya/MGlobal.h>
#include <maya/MObject.h>
#include <maya/MPxNode.h>
#include <maya/MStatus.h>

// ================================
// Plugin registration
// ================================
MStatus initializePlugin(MObject obj) {
    MStatus status;
    MFnPlugin plugin(obj, "TensifyPlugin", "1.0", "Any", &status);
    if (status != MStatus::kSuccess) {
        MGlobal::displayError("TensifyPlugin: MFnPlugin construction failed");
        return status;
    }

    // Register tensify node
    status = plugin.registerNode(
        "tensify",
        tensify::id,
        tensify::createInstance,
        tensify::initializeNode,
        MPxNode::kDependNode
    );
    if (status != MStatus::kSuccess) {
        MGlobal::displayError("TensifyPlugin: failed to register node tensify");
        return status;
    }

    // Register command
    status = plugin.registerCommand(
        "TensifyApplyColors",
        TensifyApplyColorsCreator
    );

    if (status != MStatus::kSuccess) {
        MGlobal::displayError("TensifyPlugin: failed to register command TensifyApplyColors");

        plugin.deregisterNode(tensify::id);
        return status;
    }

    MGlobal::displayInfo("TensifyPlugin: tensify node and TensifyApplyColors command registered");
    return MStatus::kSuccess;
}

MStatus uninitializePlugin(MObject obj) {
    MStatus status;
    MFnPlugin plugin(obj);

    // Deregister command first
    status = plugin.deregisterCommand("TensifyApplyColors");
    if (status != MStatus::kSuccess) {
        MGlobal::displayError("TensifyPlugin: failed to deregister command TensifyApplyColors");
        
    }

    // Deregister node
    status = plugin.deregisterNode(tensify::id);
    if (status != MStatus::kSuccess) {
        MGlobal::displayError("TensifyPlugin: failed to deregister node tensify");
    }

    // Clear global buffers to avoid leaks between sessions
    TensifyColorBuffer::clearAll();

    MGlobal::displayInfo("TensifyPlugin: unloaded, node/command deregistered and buffers cleared");
    return MStatus::kSuccess;
}
