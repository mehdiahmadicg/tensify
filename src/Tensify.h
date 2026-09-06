// Tensify.h

#ifndef TENSIFY_H
#define TENSIFY_H

#include <cstdint>
#include <map>
#include <mutex>
#include <vector>

#include <maya/MArgList.h>
#include <maya/MColor.h>
#include <maya/MColorArray.h>
#include <maya/MDataHandle.h>
#include <maya/MDGContext.h>
#include <maya/MDoubleArray.h>
#include <maya/MEvaluationNode.h>
#include <maya/MObject.h>
#include <maya/MPlug.h>
#include <maya/MPlugArray.h>
#include <maya/MPxCommand.h>
#include <maya/MPxNode.h>
#include <maya/MStatus.h>
#include <maya/MTypeId.h>


// ================================
// TensifyColorBuffer
// ================================
namespace TensifyColorBuffer {
    inline std::mutex g_mutex;
    inline std::map<int, MColorArray> g_buffers;
    inline int g_nextId = 1;

    inline int storeBuffer(const MColorArray& arr) {
        std::lock_guard<std::mutex> lk(g_mutex);
        int id = g_nextId++;
        g_buffers[id] = arr;
        return id;
    }

    inline bool fetchBuffer(int id, MColorArray& out) {
        std::lock_guard<std::mutex> lk(g_mutex);
        auto it = g_buffers.find(id);
        if (it == g_buffers.end()) return false;
        out = it->second;
        return true;
    }

    inline void eraseBuffer(int id) {
        std::lock_guard<std::mutex> lk(g_mutex);
        g_buffers.erase(id);
    }

    inline void clearAll() {
        std::lock_guard<std::mutex> lk(g_mutex);
        g_buffers.clear();
    }
}

// Creator wrapper prototype
void* TensifyApplyColorsCreator();

// Command class declaration
class TensifyApplyColorsCommand : public MPxCommand {
public:
    TensifyApplyColorsCommand();
    virtual ~TensifyApplyColorsCommand() = default;
    virtual MStatus doIt(const MArgList& args) override;
    static void* creator();
};

// Main node class
class tensify : public MPxNode {
public:
    static MTypeId id;

    static MObject clrAttribute;
    static MObject shp;
    static MObject finalShp;
    static MObject refShp;
    static MObject tensifyShader;
    static MObject stretchMap;
    static MObject compressionMap;
    static MObject smoothingEnabledAttr;
    static MObject smoothingRadiusAttr;
    static MObject smoothingThresholdAttr;

    // ------------------------------------------------------------------
    // Per-instance cached state
    // ------------------------------------------------------------------

    // cached per mesh edge lengths + dirty flags
    MDoubleArray refLen;
    MDoubleArray shpLen;
    bool refDrt = true;
    bool shpDrt = true;

    // flat CSR adjacency cache (topology)
    std::vector<int> topoRowPtr;
    std::vector<int> topoColIdx;
    int prevNumVerts = -1;
    int prevNumEdges = -1;

    // cached CSR smoothing matrix
    std::vector<int> cachedRowPtr;
    std::vector<int> cachedColIdx;
    std::vector<float> cachedVals;
    int cachedNumVerts = 0;
    uint64_t cachedDegSum = 0;
    double cachedT = -1.0;

    std::vector<float> prevX;
    std::vector<float> cg_r;
    std::vector<float> cg_p;
    std::vector<float> cg_Ap;
    std::vector<float> cg_z;

    // color ramp LUT 
    std::vector<MColor> rampLUT;
    bool rampDirty = true;

    // incremental vertex color update
    bool colorSetInitialized = false;
    std::vector<float> prevShaderFlat;

    // lifecycle
    static MStatus initializeNode();
    MStatus preEvaluation(const MDGContext& context, const MEvaluationNode& evaluationNode) override;
    static void* createInstance() { return new tensify(); }
    void postConstructor() override;

    SchedulingType schedulingType() const override { return SchedulingType::kGloballySerial; }

    // core
    MStatus compute(const MPlug& plug, MDataBlock& data) override;

    // per instance implementation
    MDoubleArray calculateLen(const MDataHandle& meshData);

    MStatus setDependentsDirty(const MPlug& dirtyPlug, MPlugArray& affectedPlugs) override;
};

#endif 
