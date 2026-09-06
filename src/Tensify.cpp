// Tensify.cpp

#include "Tensify.h"

#include <maya/MArgList.h>
#include <maya/MColorArray.h>
#include <maya/MDoubleArray.h>
#include <maya/MFn.h>
#include <maya/MFnDagNode.h>
#include <maya/MFnDoubleArrayData.h>
#include <maya/MFnMesh.h>
#include <maya/MFnMeshData.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnTypedAttribute.h>
#include <maya/MIntArray.h>
#include <maya/MItMeshEdge.h>
#include <maya/MItMeshVertex.h>
#include <maya/MPointArray.h>
#include <maya/MRampAttribute.h>
#include <maya/MSelectionList.h>
#include <maya/MString.h>
#include <maya/MStringArray.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <functional>
#include <limits>
#include <mutex>
#include <thread>
#include <tuple>
#include <vector>

#include <immintrin.h>

// chunk size
constexpr int MT_MIN_VERTS = 50000;
constexpr int MIN_PER_THREAD = 8192;
static const int S_RAMP_LUT_DEFAULT_SIZE = 1024;
constexpr int RAMP_LUT_SIZE = S_RAMP_LUT_DEFAULT_SIZE;

static unsigned int HW_THREADS = []() {
    unsigned int n = std::thread::hardware_concurrency();
    return (n == 0 ? 2 : n);
    }();


MString REF_INPUT("refInput");
MString INPUT("input");

// namespace - setColor
namespace {
    MStatus setColor(const MObject& activeObj, const MObject& colorObj, int num, float loc, const MColor& vClr, int mix) {
        MStatus status;
        MPlug colorPlg(activeObj, colorObj);
        MPlug elem = colorPlg.elementByLogicalIndex(num, &status);
        if (status != MS::kSuccess) return status;
        MPlug posPlug = elem.child(0);
        posPlug.setFloat(loc);

        MPlug colPlug = elem.child(1);
        colPlug.child(0).setFloat(vClr.r);
        colPlug.child(1).setFloat(vClr.g);
        colPlug.child(2).setFloat(vClr.b);

        MPlug mixPlug = elem.child(2);
        mixPlug.setInt(mix);

        return MS::kSuccess;
    }
}


MTypeId tensify::id(0x10000000);
MObject tensify::refShp, tensify::shp, tensify::finalShp, tensify::tensifyShader, tensify::clrAttribute;
MObject tensify::stretchMap, tensify::compressionMap;
MObject tensify::smoothingEnabledAttr;
MObject tensify::smoothingRadiusAttr;
MObject tensify::smoothingThresholdAttr;

// --- SIMD helpers (AVX2) ---
#ifdef __AVX2__
static inline double horizontal_sum_avx256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum1 = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(sum1);
    __m128 sums = _mm_add_ps(sum1, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return (double)_mm_cvtss_f32(sums);
}

// Thread pool
static std::vector<std::thread> s_tp_workers;
static std::mutex s_tp_mutex;
static std::condition_variable s_tp_cv;
static std::function<void(int, int, int)> s_tp_task;
static int s_tp_taskStart = 0;
static int s_tp_taskEnd = 0;
static int s_tp_taskThreads = 1;
static std::atomic<int> s_tp_gen(0);
static std::atomic<int> s_tp_done(0);
static bool s_tp_shutdown = false;
static int s_tp_taskPer = 0;

static void s_tp_worker_loop(int tid) {
    int lastSeenGen = 0;
    while (true) {
        std::unique_lock<std::mutex> lk(s_tp_mutex);
        s_tp_cv.wait(lk, [&] { return s_tp_shutdown || s_tp_gen.load() > lastSeenGen; });
        if (s_tp_shutdown) return;
        int gen = s_tp_gen.load();
        int start = s_tp_taskStart;
        int end = s_tp_taskEnd;
        int threads = s_tp_taskThreads;
        int per = s_tp_taskPer;
        auto func = s_tp_task;
        lk.unlock();

        int myS = start + tid * per;
        int myE = std::min(end, myS + per);
        bool participating = (tid < threads);
        if (participating) {
            if (myS < myE && func) {
                func(tid, myS, myE);
            }
            if (s_tp_done.fetch_add(1) + 1 == threads) {
                std::lock_guard<std::mutex> lk2(s_tp_mutex);
                s_tp_cv.notify_all();
            }
        }
        lastSeenGen = gen;
    }
}

static void s_tp_init_if_needed(unsigned int requestedThreads) {
    if (!s_tp_workers.empty()) return;
    unsigned int hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 2;
    unsigned int poolSize = std::max<unsigned>(1u, std::min<unsigned>(hw, std::max<unsigned>(1u, requestedThreads)));
    const unsigned int POOL_MAX = std::max<unsigned>(4u, hw * 2);
    if (poolSize > POOL_MAX) poolSize = POOL_MAX;
    s_tp_shutdown = false;
    s_tp_gen.store(0);
    s_tp_done.store(0);
    s_tp_workers.reserve(poolSize);
    for (unsigned int i = 0; i < poolSize; ++i) {
        s_tp_workers.emplace_back([i]() { s_tp_worker_loop((int)i); });
    }
}

static void s_tp_shutdown_pool() {
    {
        std::lock_guard<std::mutex> lk(s_tp_mutex);
        s_tp_shutdown = true;
    }
    s_tp_cv.notify_all();
    for (auto& th : s_tp_workers) if (th.joinable()) th.join();
    s_tp_workers.clear();
}

template<typename Fn>
static void s_tp_parallel_for(int start, int end, int threads, Fn fn) {
    if (threads <= 1 || start >= end) {
        fn(0, start, end);
        return;
    }
    s_tp_init_if_needed((unsigned)threads);
    if (s_tp_workers.empty()) {
        fn(0, start, end);
        return;
    }
    unsigned int poolSize = (unsigned)s_tp_workers.size();
    int desired = threads;
    if (desired < 1) desired = 1;
    int workerParticipants = std::min<int>((int)poolSize, std::max(0, desired - 1));

    if (workerParticipants <= 0) {
        fn(0, start, end);
        return;
    }

    int totalParticipants = workerParticipants + 1;
    int len = end - start;
    int per = (len + totalParticipants - 1) / totalParticipants;

    {
        std::lock_guard<std::mutex> lk(s_tp_mutex);
        s_tp_task = fn;
        s_tp_taskStart = start;
        s_tp_taskEnd = end;
        s_tp_taskThreads = workerParticipants;
        s_tp_taskPer = per;
        s_tp_done.store(0);
        s_tp_gen.fetch_add(1);
    }
    s_tp_cv.notify_all();

    int mainTid = workerParticipants;
    int mainS = start + mainTid * per;
    int mainE = std::min(end, mainS + per);
    if (mainS < mainE) {
        fn(mainTid, mainS, mainE);
    }

    std::unique_lock<std::mutex> lk(s_tp_mutex);
    s_tp_cv.wait(lk, [&] { return s_tp_done.load() >= workerParticipants; });
}

// AVX Math Helpers
static void csr_matvec_avx(const std::vector<int>& rowPtr, const std::vector<int>& colIdx,
    const std::vector<float>& vals, const float* xin, float* out, int numVerts) {
    unsigned int hw = HW_THREADS;
    int threads = 1;
    if (numVerts >= 4096) threads = std::min<unsigned>(hw, (unsigned)std::max(1, numVerts / MIN_PER_THREAD));

    if (threads <= 1) {
        for (int i = 0; i < numVerts; ++i) {
            int r0 = rowPtr[i], r1 = rowPtr[i + 1];
            int k = r0;
            double accd = 0.0;
            for (; k + 7 < r1; k += 8) {
                if (k + 8 < r1) {
                    int pi = colIdx[k + 8];
                    _mm_prefetch((const char*)&xin[pi], _MM_HINT_T0);
                }
                __m256i idx = _mm256_setr_epi32(colIdx[k], colIdx[k + 1], colIdx[k + 2], colIdx[k + 3],
                    colIdx[k + 4], colIdx[k + 5], colIdx[k + 6], colIdx[k + 7]);
                __m256 xvec = _mm256_i32gather_ps(xin, idx, 4);
                __m256 vvec = _mm256_loadu_ps(&vals[k]);
                __m256 prod = _mm256_mul_ps(vvec, xvec);
                accd += horizontal_sum_avx256(prod);
            }
            for (; k < r1; ++k) accd += (double)vals[k] * (double)xin[colIdx[k]];
            out[i] = (float)accd;
        }
        return;
    }
    // Parallel implementation
    s_tp_parallel_for(0, numVerts, threads, [&](int tid, int s, int e) {
        for (int i = s; i < e; ++i) {
            int r0 = rowPtr[i], r1 = rowPtr[i + 1];
            int k = r0;
            double accd = 0.0;
            for (; k + 7 < r1; k += 8) {
                if (k + 8 < r1) {
                    int pi = colIdx[k + 8];
                    _mm_prefetch((const char*)&xin[pi], _MM_HINT_T0);
                }
                __m256i idx = _mm256_setr_epi32(colIdx[k], colIdx[k + 1], colIdx[k + 2], colIdx[k + 3],
                    colIdx[k + 4], colIdx[k + 5], colIdx[k + 6], colIdx[k + 7]);
                __m256 xvec = _mm256_i32gather_ps(xin, idx, 4);
                __m256 vvec = _mm256_loadu_ps(&vals[k]);
                __m256 prod = _mm256_mul_ps(vvec, xvec);
                accd += horizontal_sum_avx256(prod);
            }
            for (; k < r1; ++k) accd += (double)vals[k] * (double)xin[colIdx[k]];
            out[i] = (float)accd;
        }
        });
}

static double simd_dot_double(const float* a, const float* b, int n) {
    unsigned int hwLocal = HW_THREADS;
    int threads = 1;
    if (n >= 4096) threads = std::min<unsigned>(hwLocal, (unsigned)std::max(1, n / MIN_PER_THREAD));
    if (threads <= 1) {
        int i = 0;
        double acc = 0.0;
        for (; i + 7 < n; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            __m256 prod = _mm256_mul_ps(va, vb);
            acc += horizontal_sum_avx256(prod);
        }
        for (; i < n; ++i) acc += (double)a[i] * (double)b[i];
        return acc;
    }
    std::vector<double> partial(threads, 0.0);
    s_tp_parallel_for(0, n, threads, [&](int tid, int s, int e) {
        int i = s;
        double acc = 0.0;
        for (; i + 7 < e; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            __m256 prod = _mm256_mul_ps(va, vb);
            acc += horizontal_sum_avx256(prod);
        }
        for (; i < e; ++i) acc += (double)a[i] * (double)b[i];
        partial[tid] = acc;
        });
    double total = 0.0;
    for (double v : partial) total += v;
    return total;
}

static void simd_axpy_inplace(float* x, const float* p, float alpha, int n) {
    unsigned int hwLocal = HW_THREADS;
    int threads = 1;
    if (n >= 4096) threads = std::min<unsigned>(hwLocal, (unsigned)std::max(1, n / MIN_PER_THREAD));
    if (threads <= 1) {
        int i = 0;
        __m256 a = _mm256_set1_ps(alpha);
        for (; i + 7 < n; i += 8) {
            __m256 vx = _mm256_loadu_ps(x + i);
            __m256 vp = _mm256_loadu_ps(p + i);
            _mm256_storeu_ps(x + i, _mm256_add_ps(vx, _mm256_mul_ps(a, vp)));
        }
        for (; i < n; ++i) x[i] += alpha * p[i];
        return;
    }
    s_tp_parallel_for(0, n, threads, [&](int tid, int s, int e) {
        int i = s;
        __m256 a = _mm256_set1_ps(alpha);
        for (; i + 7 < e; i += 8) {
            __m256 vx = _mm256_loadu_ps(x + i);
            __m256 vp = _mm256_loadu_ps(p + i);
            _mm256_storeu_ps(x + i, _mm256_add_ps(vx, _mm256_mul_ps(a, vp)));
        }
        for (; i < e; ++i) x[i] += alpha * p[i];
        });
}

static void simd_axpy_sub_inplace(float* r, const float* Ap, float alpha, int n) {
    unsigned int hwLocal = HW_THREADS;
    int threads = 1;
    if (n >= 4096) threads = std::min<unsigned>(hwLocal, (unsigned)std::max(1, n / MIN_PER_THREAD));
    if (threads <= 1) {
        int i = 0;
        __m256 a = _mm256_set1_ps(alpha);
        for (; i + 7 < n; i += 8) {
            __m256 vr = _mm256_loadu_ps(r + i);
            __m256 vAp = _mm256_loadu_ps(Ap + i);
            _mm256_storeu_ps(r + i, _mm256_sub_ps(vr, _mm256_mul_ps(a, vAp)));
        }
        for (; i < n; ++i) r[i] -= alpha * Ap[i];
        return;
    }
    s_tp_parallel_for(0, n, threads, [&](int tid, int s, int e) {
        int i = s;
        __m256 a = _mm256_set1_ps(alpha);
        for (; i + 7 < e; i += 8) {
            __m256 vr = _mm256_loadu_ps(r + i);
            __m256 vAp = _mm256_loadu_ps(Ap + i);
            _mm256_storeu_ps(r + i, _mm256_sub_ps(vr, _mm256_mul_ps(a, vAp)));
        }
        for (; i < e; ++i) r[i] -= alpha * Ap[i];
        });
}

static double simd_rnorm_double(const float* r, int n) {
    unsigned int hwLocal = HW_THREADS;
    int threads = 1;
    if (n >= 4096) threads = std::min<unsigned>(hwLocal, (unsigned)std::max(1, n / MIN_PER_THREAD));
    if (threads <= 1) {
        int i = 0;
        double acc = 0.0;
        for (; i + 7 < n; i += 8) {
            __m256 vr = _mm256_loadu_ps(r + i);
            __m256 prod = _mm256_mul_ps(vr, vr);
            acc += horizontal_sum_avx256(prod);
        }
        for (; i < n; ++i) acc += (double)r[i] * (double)r[i];
        return acc;
    }
    std::vector<double> partial(threads, 0.0);
    s_tp_parallel_for(0, n, threads, [&](int tid, int s, int e) {
        int i = s;
        double acc = 0.0;
        for (; i + 7 < e; i += 8) {
            __m256 vr = _mm256_loadu_ps(r + i);
            __m256 prod = _mm256_mul_ps(vr, vr);
            acc += horizontal_sum_avx256(prod);
        }
        for (; i < e; ++i) acc += (double)r[i] * (double)r[i];
        partial[tid] = acc;
        });
    double total = 0.0;
    for (double v : partial) total += v;
    return total;
}

static void simd_div_inplace(float* out, const float* num, const float* den, int n) {
    unsigned int hwLocal = HW_THREADS;
    int threads = 1;
    if (n >= 4096) threads = std::min<unsigned>(hwLocal, (unsigned)std::max(1, n / MIN_PER_THREAD));
    if (threads <= 1) {
        int i = 0;
        for (; i + 7 < n; i += 8) {
            _mm256_storeu_ps(out + i, _mm256_div_ps(_mm256_loadu_ps(num + i), _mm256_loadu_ps(den + i)));
        }
        for (; i < n; ++i) out[i] = num[i] / den[i];
        return;
    }
    s_tp_parallel_for(0, n, threads, [&](int tid, int s, int e) {
        int i = s;
        for (; i + 7 < e; i += 8) {
            _mm256_storeu_ps(out + i, _mm256_div_ps(_mm256_loadu_ps(num + i), _mm256_loadu_ps(den + i)));
        }
        for (; i < e; ++i) out[i] = num[i] / den[i];
        });
}

static void simd_p_update(float* p, const float* z, double beta, int n) {
    unsigned int hwLocal = HW_THREADS;
    int threads = 1;
    if (n >= 4096) threads = std::min<unsigned>(hwLocal, (unsigned)std::max(1, n / MIN_PER_THREAD));
    if (threads <= 1) {
        int i = 0;
        __m256 vb = _mm256_set1_ps((float)beta);
        for (; i + 7 < n; i += 8) {
            __m256 vz = _mm256_loadu_ps(z + i);
            __m256 vp = _mm256_loadu_ps(p + i);
            _mm256_storeu_ps(p + i, _mm256_add_ps(vz, _mm256_mul_ps(vb, vp)));
        }
        for (; i < n; ++i) p[i] = z[i] + (float)(beta * (double)p[i]);
        return;
    }
    s_tp_parallel_for(0, n, threads, [&](int tid, int s, int e) {
        int i = s;
        __m256 vb = _mm256_set1_ps((float)beta);
        for (; i + 7 < e; i += 8) {
            __m256 vz = _mm256_loadu_ps(z + i);
            __m256 vp = _mm256_loadu_ps(p + i);
            _mm256_storeu_ps(p + i, _mm256_add_ps(vz, _mm256_mul_ps(vb, vp)));
        }
        for (; i < e; ++i) p[i] = z[i] + (float)(beta * (double)p[i]);
        });
}

#else
// Fallback 
#endif


namespace {
    thread_local MColorArray s_subsetColors;
    thread_local MColorArray s_chunkColors;
    thread_local MIntArray   s_chunkIndices;

    inline void ensureLength(MColorArray& arr, int need) {
        if ((int)arr.length() < need) arr.setLength(need);
    }
    inline void ensureLength(MIntArray& arr, int need) {
        if ((int)arr.length() < need) arr.setLength(need);
    }
}

namespace {
    const int _i1 = 0; const float _p1 = 0.0f; const MColor _c1(0, 1, 0);
    const int _i2 = 1; const float _p2 = 0.5f; const MColor _c2(0, 0, 0);
    const int _i3 = 2; const float _p3 = 1.0f; const MColor _c3(1, 0, 0);
}

TensifyApplyColorsCommand::TensifyApplyColorsCommand() {}

MStatus TensifyApplyColorsCommand::doIt(const MArgList& args) {
    if (args.length() < 2) return MS::kFailure;
    MString shapeName = args.asString(0);
    int bufferId = args.asInt(1);
    int start = 0;
    int length = -1;
    if (args.length() >= 3) start = args.asInt(2);
    if (args.length() >= 4) length = args.asInt(3);

    MColorArray colors;
    if (!TensifyColorBuffer::fetchBuffer(bufferId, colors)) return MS::kFailure;

    int total = (int)colors.length();
    if (length < 0) length = total - start;
    if (start < 0 || start >= total) return MS::kFailure;
    if (start + length > total) length = total - start;

    MSelectionList sel;
    sel.add(shapeName);
    MObject shapeObj;
    sel.getDependNode(0, shapeObj);
    if (shapeObj.isNull()) return MS::kFailure;

    MFnMesh fnMesh(shapeObj);
    MColorArray chunkColors;
    MIntArray indices;
    chunkColors.setLength(length);
    indices.setLength(length);
    for (int i = 0; i < length; ++i) {
        chunkColors[i] = colors[start + i];
        indices[i] = start + i;
    }

    MStatus st = fnMesh.setVertexColors(chunkColors, indices);
    if (st != MS::kSuccess) return st;

    if (start + length >= total) {
        TensifyColorBuffer::eraseBuffer(bufferId);
    }

    return MS::kSuccess;
}

void* TensifyApplyColorsCommand::creator() { return new TensifyApplyColorsCommand(); }
void* TensifyApplyColorsCreator() { return TensifyApplyColorsCommand::creator(); }

MStatus tensify::initializeNode() {
    MFnTypedAttribute maker;
    refShp = maker.create(REF_INPUT, REF_INPUT, MFnMeshData::kMesh); maker.setStorable(true);
    shp = maker.create(INPUT, INPUT, MFnMeshData::kMesh); maker.setStorable(true);
    finalShp = maker.create("output", "output", MFnMeshData::kMesh); maker.setWritable(false); maker.setStorable(false);
    clrAttribute = MRampAttribute::createColorRamp("tensifyColor", "tensifyColor");
    addAttribute(refShp); addAttribute(shp); addAttribute(finalShp); addAttribute(clrAttribute);
    attributeAffects(refShp, finalShp); attributeAffects(shp, finalShp); attributeAffects(clrAttribute, finalShp);

    MFnNumericAttribute nAttr;
    tensifyShader = nAttr.create("tensifyShader", "ts", MFnNumericData::kBoolean, false);
    nAttr.setKeyable(true); nAttr.setStorable(true);
    addAttribute(tensifyShader);

    MFnTypedAttribute tAttr;
    stretchMap = tAttr.create("stretchMap", "stretchMap", MFnData::kDoubleArray);
    tAttr.setStorable(false); tAttr.setWritable(false);
    addAttribute(stretchMap);
    attributeAffects(refShp, stretchMap); attributeAffects(shp, stretchMap);

    compressionMap = tAttr.create("compressionMap", "compressionMap", MFnData::kDoubleArray);
    tAttr.setStorable(false); tAttr.setWritable(false);
    addAttribute(compressionMap);
    attributeAffects(refShp, compressionMap); attributeAffects(shp, compressionMap);

    smoothingEnabledAttr = nAttr.create("smoothingEnabled", "smo_en", MFnNumericData::kBoolean, true);
    nAttr.setStorable(true); nAttr.setKeyable(false); nAttr.setHidden(true);
    addAttribute(smoothingEnabledAttr);
    attributeAffects(smoothingEnabledAttr, finalShp);
    attributeAffects(smoothingEnabledAttr, stretchMap);
    attributeAffects(smoothingEnabledAttr, compressionMap);

    smoothingRadiusAttr = nAttr.create("smoothingRadius", "smo_rad", MFnNumericData::kInt, 0);
    nAttr.setStorable(true); nAttr.setKeyable(true); nAttr.setMin(0);
    nAttr.setMax(std::numeric_limits<int>::max());
    addAttribute(smoothingRadiusAttr);
    attributeAffects(smoothingRadiusAttr, finalShp);
    attributeAffects(smoothingRadiusAttr, stretchMap);
    attributeAffects(smoothingRadiusAttr, compressionMap);

    smoothingThresholdAttr = nAttr.create("smoothingThreshold", "smo_thr", MFnNumericData::kDouble, 0.0);
    nAttr.setStorable(true); nAttr.setKeyable(false); nAttr.setWritable(false);
    addAttribute(smoothingThresholdAttr);
    attributeAffects(smoothingThresholdAttr, finalShp);
    attributeAffects(smoothingThresholdAttr, stretchMap);
    attributeAffects(smoothingThresholdAttr, compressionMap);

    return MStatus::kSuccess;
}

void tensify::postConstructor() {
    std::vector<std::tuple<int, float, MColor>> rampValues = {
        {_i1, _p1, _c1}, {_i2, _p2, _c2}, {_i3, _p3, _c3}
    };
    for (const auto& ramp : rampValues) {
        setColor(thisMObject(), clrAttribute, std::get<0>(ramp), std::get<1>(ramp), std::get<2>(ramp), 1);
    }
}

MDoubleArray tensify::calculateLen(const MDataHandle& dataSource) {
    MStatus status;
    MObject meshObj = dataSource.asMesh();
    MDoubleArray meanSqRecords;
    if (meshObj.isNull()) return meanSqRecords;

    MFnMesh fn(meshObj, &status);
    if (status != MS::kSuccess) return meanSqRecords;

    int numVerts = fn.numVertices(&status);
    if (status != MS::kSuccess || numVerts <= 0) return meanSqRecords;

    MPointArray points;
    fn.getPoints(points, MSpace::kObject);

    // Access raw double pointer 
    const double* rawPoints = (const double*)&points[0].x;

    meanSqRecords.setLength(numVerts);

    int numEdges = fn.numEdges(&status);

    bool topoChanged = ((int)topoRowPtr.size() != numVerts + 1) || (prevNumVerts != numVerts) || (prevNumEdges != numEdges);

    if (topoChanged) {
        // Build flat CSR topology once, reuse for everything.
        topoRowPtr.assign(numVerts + 1, 0);
        std::vector<std::vector<int>> tempAdj(numVerts);

        MItMeshEdge eIt(meshObj, &status);
        if (status == MS::kSuccess) {
            for (eIt.reset(); !eIt.isDone(); eIt.next()) {
                int v1 = eIt.index(0);
                int v2 = eIt.index(1);
                if (v1 >= 0 && v1 < numVerts && v2 >= 0 && v2 < numVerts) {
                    // Only add if not duplicate (though edge iterator implies unique edges)
                    bool f1 = false; for (int k : tempAdj[v1]) if (k == v2) { f1 = true; break; }
                    if (!f1) tempAdj[v1].push_back(v2);

                    bool f2 = false; for (int k : tempAdj[v2]) if (k == v1) { f2 = true; break; }
                    if (!f2) tempAdj[v2].push_back(v1);
                }
            }
        }
        else {
            // Fallback
            MItMeshVertex vIt(meshObj, &status);
            if (status == MS::kSuccess) {
                for (vIt.reset(); !vIt.isDone(); vIt.next()) {
                    MIntArray conn; vIt.getConnectedVertices(conn);
                    int idx = vIt.index();
                    for (unsigned k = 0; k < conn.length(); ++k) tempAdj[idx].push_back(conn[k]);
                }
            }
        }

        // Flatten to CSR
        int accumulated = 0;
        for (int i = 0; i < numVerts; ++i) {
            topoRowPtr[i] = accumulated;
            accumulated += (int)tempAdj[i].size();
        }
        topoRowPtr[numVerts] = accumulated;
        topoColIdx.resize(accumulated);

        int offset = 0;
        for (int i = 0; i < numVerts; ++i) {
            for (int nb : tempAdj[i]) {
                topoColIdx[offset++] = nb;
            }
        }

        prevNumVerts = numVerts;
        prevNumEdges = numEdges;
    }

    unsigned int hwThreads = HW_THREADS;
    int compThreads = 1;
    if (numVerts >= 4096) {
        compThreads = std::min<unsigned>(hwThreads, std::max(1, numVerts / MIN_PER_THREAD));
    }

    // Use thread pool with optimized raw pointer access and CSR traversal
    s_tp_parallel_for(0, numVerts, compThreads, [&](int tid, int s, int e) {
        for (int v = s; v < e; ++v) {
            int r0 = topoRowPtr[v];
            int r1 = topoRowPtr[v + 1];
            int nlen = r1 - r0;

            if (nlen == 0) {
                meanSqRecords[v] = 0.0;
                continue;
            }

            // Direct memory access for vertex
            double v_x = rawPoints[v * 4];
            double v_y = rawPoints[v * 4 + 1];
            double v_z = rawPoints[v * 4 + 2];

            double sumSq = 0.0;
            for (int k = r0; k < r1; ++k) {
                int nid = topoColIdx[k];
                // Direct memory access for neighbor
                double n_x = rawPoints[nid * 4];
                double n_y = rawPoints[nid * 4 + 1];
                double n_z = rawPoints[nid * 4 + 2];

                double dx = v_x - n_x;
                double dy = v_y - n_y;
                double dz = v_z - n_z;
                sumSq += std::sqrt(dx * dx + dy * dy + dz * dz);
            }
            meanSqRecords[v] = sumSq / (double)nlen;
        }
        });

    return meanSqRecords;
}

MStatus tensify::compute(const MPlug& plug, MDataBlock& data) {
    MStatus status;
    if (plug != finalShp && plug != stretchMap && plug != compressionMap) return MStatus::kUnknownParameter;

    auto refGeoData = data.inputValue(refShp, &status);
    auto meshState = data.inputValue(shp, &status);
    auto resultGeo = data.outputValue(finalShp, &status);
    MRampAttribute colorAttribute(thisMObject(), clrAttribute, &status);

    if (refLen.length() == 0 || refDrt) {
        refLen = calculateLen(refGeoData);
        refDrt = false;
    }
    if (shpLen.length() == 0 || shpDrt) {
        shpLen = calculateLen(meshState);
        shpDrt = false;
    }

    resultGeo.copy(meshState);
    resultGeo.set(meshState.asMesh());
    MFnMesh meshFnOut(resultGeo.asMesh(), &status);

    MFnMesh meshFnScene;
    bool haveSceneMesh = false;
    MPlug shpPlug(thisMObject(), shp);
    MPlugArray connections;
    if (!shpPlug.isNull()) {
        shpPlug.connectedTo(connections, false, true);
        for (unsigned ci = 0; ci < connections.length(); ++ci) {
            MPlug connPlug = connections[ci];
            MObject nodeObj = connPlug.node();
            if (nodeObj.isNull()) continue;
            if (nodeObj.hasFn(MFn::kMesh)) { meshFnScene.setObject(nodeObj); if (!meshFnScene.object().isNull()) { haveSceneMesh = true; break; } }
            if (nodeObj.hasFn(MFn::kTransform) || nodeObj.hasFn(MFn::kDagNode)) {
                MFnDagNode dag(nodeObj);
                for (unsigned ci2 = 0; ci2 < (unsigned)dag.childCount(); ++ci2) {
                    MObject child = dag.child(ci2);
                    if (child.isNull()) continue;
                    if (child.hasFn(MFn::kMesh)) { meshFnScene.setObject(child); if (!meshFnScene.object().isNull()) { haveSceneMesh = true; break; } }
                }
                if (haveSceneMesh) break;
            }
        }
    }
    if (!haveSceneMesh) {
        MPlug outPlug(thisMObject(), finalShp);
        MPlugArray outConnections;
        if (!outPlug.isNull()) {
            outPlug.connectedTo(outConnections, true, false);
            for (unsigned oi = 0; oi < outConnections.length(); ++oi) {
                MPlug dstPlug = outConnections[oi];
                MObject dstNode = dstPlug.node();
                if (dstNode.isNull()) continue;
                if (dstNode.hasFn(MFn::kMesh)) { meshFnScene.setObject(dstNode); if (!meshFnScene.object().isNull()) { haveSceneMesh = true; break; } }
                if (dstNode.hasFn(MFn::kTransform) || dstNode.hasFn(MFn::kDagNode)) {
                    MFnDagNode dag(dstNode);
                    for (unsigned di = 0; di < (unsigned)dag.childCount(); ++di) {
                        MObject child = dag.child(di);
                        if (child.isNull()) continue;
                        if (child.hasFn(MFn::kMesh)) { meshFnScene.setObject(child); if (!meshFnScene.object().isNull()) { haveSceneMesh = true; break; } }
                    }
                    if (haveSceneMesh) break;
                }
            }
        }
    }

    MFnMesh& meshFn = (haveSceneMesh ? meshFnScene : meshFnOut);

    MStatus st;
    MColorArray shader(meshFn.numVertices(&st));
    MIntArray geoPoint(shader.length());
    MDoubleArray stretchArray, compressionArray;

    int smoothingRadiusLocal = 0;
    if (!thisMObject().isNull()) {
        MPlug pRadius(thisMObject(), smoothingRadiusAttr);
        if (!pRadius.isNull()) smoothingRadiusLocal = data.inputValue(smoothingRadiusAttr).asInt();
        if (smoothingRadiusLocal < 0) smoothingRadiusLocal = 0;
    }

    MDoubleArray rawRatios;
    int numVerts = shader.length();
    rawRatios.setLength(numVerts);
    unsigned int hw = HW_THREADS;

    // Vectorized AVX2 + Thread Pool
    int numThreads = 1;
    if (numVerts >= 4096) numThreads = std::min<unsigned>(hw, std::max(1, numVerts / MIN_PER_THREAD));

    s_tp_parallel_for(0, numVerts, numThreads, [&](int tid, int s, int e) {
        const double* pRef = (refLen.length() > 0) ? &refLen[0] : nullptr;
        const double* pShp = (shpLen.length() > 0) ? &shpLen[0] : nullptr;
        bool safe = (pRef && pShp && (int)refLen.length() >= numVerts && (int)shpLen.length() >= numVerts);

        int i = s;
        if (safe) {
            // Vectorized loop for double precision
#ifdef __AVX2__
            __m256d vOne = _mm256_set1_pd(1.0);
            __m256d vHalf = _mm256_set1_pd(0.5);
            __m256d vZero = _mm256_setzero_pd();
            __m256d vEps = _mm256_set1_pd(1e-12);

            for (; i + 3 < e; i += 4) {
                __m256d refSq = _mm256_loadu_pd(pRef + i);
                __m256d shpSq = _mm256_loadu_pd(pShp + i);
                __m256d refVal = _mm256_sqrt_pd(refSq);
                __m256d shpVal = _mm256_sqrt_pd(_mm256_max_pd(vZero, shpSq));
                __m256d mask = _mm256_cmp_pd(refSq, vEps, _CMP_GT_OQ);
                // calc (ref-shp)/ref + 0.5
                __m256d diff = _mm256_sub_pd(refVal, shpVal);
                __m256d ratio = _mm256_div_pd(diff, refVal);
                ratio = _mm256_add_pd(ratio, vHalf);
                // blend: if refSq <= 1e-12, ratio = 0.5
                __m256d res = _mm256_blendv_pd(vHalf, ratio, mask);
                // clamp 0..1
                res = _mm256_max_pd(vZero, _mm256_min_pd(vOne, res));
                _mm256_storeu_pd(&rawRatios[i], res);
            }
#endif
            for (; i < e; ++i) {
                double refSq = pRef[i];
                if (refSq > 1e-12) {
                    double refVal = std::sqrt(refSq);
                    double shpVal = std::sqrt(std::max(0.0, pShp[i]));
                    double val = ((refVal - shpVal) / refVal) + 0.5;
                    rawRatios[i] = std::max(0.0, std::min(1.0, val));
                }
                else rawRatios[i] = 0.5;
            }
        }
        else {
            for (; i < e; ++i) rawRatios[i] = 0.5;
        }
        });

    // Build LUT
    if (rampLUT.size() != (size_t)RAMP_LUT_SIZE) { rampLUT.clear(); rampLUT.resize(RAMP_LUT_SIZE); rampDirty = true; }
    if (rampDirty) {
        const float BRIGHTNESS_BOOST = 2.0f;
        for (int ri = 0; ri < RAMP_LUT_SIZE; ++ri) {
            float pos = float(ri) / float(RAMP_LUT_SIZE - 1);
            MColor c; MStatus stc; colorAttribute.getColorAtPosition(pos, c, &stc);
            if (stc != MS::kSuccess) c = MColor(0, 0, 0);
            c.r = std::min(1.0f, c.r * BRIGHTNESS_BOOST);
            c.g = std::min(1.0f, c.g * BRIGHTNESS_BOOST);
            c.b = std::min(1.0f, c.b * BRIGHTNESS_BOOST);
            rampLUT[ri] = c;
        }
        rampDirty = false;
    }

    // Smoothing
    MDoubleArray finalRatios = rawRatios;
    if (smoothingRadiusLocal > 0 && numVerts > 0) {

        // Rebuild adjacency vectors for CSR 
        uint64_t degSum = 0;
        degSum = topoRowPtr.size() > 0 ? topoRowPtr.back() : 0;

        const double t_scale = 0.2;
        const double t = std::max(1.0, (double)smoothingRadiusLocal) * t_scale;

        bool needRebuildCSR = (cachedNumVerts != numVerts) || (cachedDegSum != degSum) || (std::fabs(cachedT - t) > 1e-12);

        if (needRebuildCSR) {
            // Build smoothing matrix directly from s_topo arrays
            cachedRowPtr.resize(numVerts + 1);
            cachedRowPtr[0] = 0;
            // First pass: Calculate row pointers
            for (int i = 0; i < numVerts; ++i) {
                int deg = topoRowPtr[i + 1] - topoRowPtr[i];
                cachedRowPtr[i + 1] = cachedRowPtr[i] + 1 + deg;
            }

            int NNZ = cachedRowPtr.back();
            cachedColIdx.resize(NNZ);
            cachedVals.resize(NNZ);

            // Parallel build of values
            int buildThreads = std::min<unsigned>(hw, std::max(1, numVerts / MIN_PER_THREAD));
            s_tp_parallel_for(0, numVerts, buildThreads, [&](int tid, int s, int e) {
                for (int i = s; i < e; ++i) {
                    int src_r0 = topoRowPtr[i];
                    int src_r1 = topoRowPtr[i + 1];
                    int deg = src_r1 - src_r0;

                    int dst_idx = cachedRowPtr[i];

                    // Diagonal
                    float diag = static_cast<float>(1.0 + t * (double)deg);
                    cachedColIdx[dst_idx] = i;
                    cachedVals[dst_idx] = diag;
                    dst_idx++;

                    // Neighbors
                    for (int k = src_r0; k < src_r1; ++k) {
                        cachedColIdx[dst_idx] = topoColIdx[k];
                        cachedVals[dst_idx] = static_cast<float>(-t);
                        dst_idx++;
                    }
                }
                });

            cachedNumVerts = numVerts; cachedDegSum = degSum; cachedT = t;
        }

        if ((int)prevX.size() != numVerts) prevX.assign(numVerts, 0.0f);
        if ((int)cg_r.size() != numVerts) cg_r.resize(numVerts);
        if ((int)cg_p.size() != numVerts) cg_p.resize(numVerts);
        if ((int)cg_Ap.size() != numVerts) cg_Ap.resize(numVerts);
        if ((int)cg_z.size() != numVerts) cg_z.resize(numVerts);

        std::vector<float> b(numVerts);
        for (int i = 0; i < numVerts; ++i) b[i] = static_cast<float>(rawRatios[i]);
        std::vector<float> x = prevX; // Start with previous frame's solution

        std::vector<float> diagA(numVerts, 1.0f);
        // Quick diag extraction
        for (int i = 0; i < numVerts; ++i) {
            diagA[i] = cachedVals[cachedRowPtr[i]];
        }

        csr_matvec_avx(cachedRowPtr, cachedColIdx, cachedVals, x.data(), cg_Ap.data(), numVerts);
        for (int i = 0; i < numVerts; ++i) cg_r[i] = b[i] - cg_Ap[i];
        simd_div_inplace(cg_z.data(), cg_r.data(), diagA.data(), numVerts);
        for (int i = 0; i < numVerts; ++i) cg_p[i] = cg_z[i];
        double rz_old = simd_dot_double(cg_r.data(), cg_z.data(), numVerts);

        const int MAX_CG_ITERS = std::min<int>(120, std::max(10, numVerts / 40));
        for (int cgIter = 0; cgIter < MAX_CG_ITERS; ++cgIter) {
            csr_matvec_avx(cachedRowPtr, cachedColIdx, cachedVals, cg_p.data(), cg_Ap.data(), numVerts);
            double pAp = simd_dot_double(cg_p.data(), cg_Ap.data(), numVerts);
            if (std::abs(pAp) < 1e-24) break;
            double alpha = rz_old / pAp;
            simd_axpy_inplace(x.data(), cg_p.data(), (float)alpha, numVerts);
            simd_axpy_sub_inplace(cg_r.data(), cg_Ap.data(), (float)alpha, numVerts);
            if (std::sqrt(simd_rnorm_double(cg_r.data(), numVerts)) < 1e-3) break;
            simd_div_inplace(cg_z.data(), cg_r.data(), diagA.data(), numVerts);
            double rz_new = simd_dot_double(cg_r.data(), cg_z.data(), numVerts);
            simd_p_update(cg_p.data(), cg_z.data(), rz_new / rz_old, numVerts);
            rz_old = rz_new;
        }
        for (int i = 0; i < numVerts; ++i) finalRatios[i] = static_cast<double>(x[i]);
        prevX = x;
    }

    int N = (int)finalRatios.length();
    shader.setLength(N); geoPoint.setLength(N);
    int mapThreads = (N >= 4096) ? std::min<unsigned>(hw, (unsigned)std::max(1, N / MIN_PER_THREAD)) : 1;

    // Use thread pool for mapping
    s_tp_parallel_for(0, N, mapThreads, [&](int tid, int s, int e) {
        for (int i = s; i < e; ++i) {
            int li = (int)std::round(finalRatios[i] * (RAMP_LUT_SIZE - 1));
            li = std::max(0, std::min(RAMP_LUT_SIZE - 1, li));
            MColor c = rampLUT[li]; c.a = 1.0f;
            shader[i] = c; geoPoint[i] = i;
        }
        });

    // Prepare outputs (Stretch/Compression) - Vectorized
    stretchArray.setLength(N); compressionArray.setLength(N);
    s_tp_parallel_for(0, N, mapThreads, [&](int tid, int s, int e) {
        int i = s;
#ifdef __AVX2__
        __m256d vHalf = _mm256_set1_pd(0.5);
        __m256d vTwo = _mm256_set1_pd(2.0);
        __m256d vZero = _mm256_setzero_pd();
        for (; i + 3 < e; i += 4) {
            __m256d ratio = _mm256_loadu_pd(&finalRatios[i]);
            // Stretch: (ratio < 0.5) ? (0.5 - ratio)*2 : 0
            __m256d diffS = _mm256_sub_pd(vHalf, ratio);
            __m256d maskS = _mm256_cmp_pd(ratio, vHalf, _CMP_LT_OQ);
            __m256d resS = _mm256_mul_pd(diffS, vTwo);
            resS = _mm256_blendv_pd(vZero, resS, maskS);
            _mm256_storeu_pd(&stretchArray[i], resS);

            // Comp: (ratio > 0.5) ? (ratio - 0.5)*2 : 0
            __m256d diffC = _mm256_sub_pd(ratio, vHalf);
            __m256d maskC = _mm256_cmp_pd(ratio, vHalf, _CMP_GT_OQ);
            __m256d resC = _mm256_mul_pd(diffC, vTwo);
            resC = _mm256_blendv_pd(vZero, resC, maskC);
            _mm256_storeu_pd(&compressionArray[i], resC);
        }
#endif
        for (; i < e; ++i) {
            double r = finalRatios[i];
            stretchArray[i] = (r < 0.5) ? (0.5 - r) * 2.0 : 0.0;
            compressionArray[i] = (r > 0.5) ? (r - 0.5) * 2.0 : 0.0;
        }
        });

    // Apply Colors
    bool shaderEnabled = false;
    MPlug pTs(thisMObject(), tensifyShader);
    if (!pTs.isNull()) shaderEnabled = data.inputValue(tensifyShader).asBool();

    if (shaderEnabled) {
        const MString colorSetName = "tensifyColors";
        MFnMesh& applyFn = haveSceneMesh ? meshFnScene : meshFnOut;

        MStringArray existingSets; applyFn.getColorSetNames(existingSets);
        bool haveSet = false;
        for (unsigned i = 0; i < existingSets.length(); ++i) if (existingSets[i] == colorSetName) { haveSet = true; break; }
        if (!haveSet) applyFn.createColorSetWithName(colorSetName);
        applyFn.setCurrentColorSetName(colorSetName);
        applyFn.setDisplayColors(true);

        int existingColorCount = 0;
        { MColorArray tmp; applyFn.getVertexColors(tmp); existingColorCount = (int)tmp.length(); }

        // colorSetInitialized
        bool needInitialFull = (!colorSetInitialized) || ((int)geoPoint.length() != (int)shader.length()) || (existingColorCount < (int)shader.length());

        int nFloats = N * 4;
        if ((int)prevShaderFlat.size() != nFloats) {
            prevShaderFlat.assign(nFloats, -1.0f);
        }

        // Parallel diff check
        std::vector<std::vector<int>> tlsChanges(hw);

        std::atomic<bool> anyChange(false);
        int diffThreads = mapThreads;

        s_tp_parallel_for(0, N, diffThreads, [&](int tid, int s, int e) {
            std::vector<int>& localChg = tlsChanges[tid];
            localChg.clear();
            const float EPS = 1e-4f; // slightly loose tolerance for float comparison
            for (int i = s; i < e; ++i) {
                const MColor& c = shader[i];
                float* prev = &prevShaderFlat[i * 4];
                bool diff = (std::fabs(c.r - prev[0]) > EPS) || (std::fabs(c.g - prev[1]) > EPS) ||
                    (std::fabs(c.b - prev[2]) > EPS);
                if (diff) {
                    localChg.push_back(i);
                    prev[0] = c.r; prev[1] = c.g; prev[2] = c.b; prev[3] = c.a;
                }
            }
            if (!localChg.empty()) anyChange = true;
            });

        MIntArray changedIndices;
        if (anyChange) {
            int totalChg = 0;
            for (int t = 0; t < diffThreads; ++t) totalChg += (int)tlsChanges[t].size();
            changedIndices.setLength(totalChg);
            int offset = 0;
            for (int t = 0; t < diffThreads; ++t) {
                const auto& vec = tlsChanges[t];
                for (int idx : vec) changedIndices[offset++] = idx;
            }
        }

        if (needInitialFull) {
            if (applyFn.setVertexColors(shader, geoPoint) == MS::kSuccess) colorSetInitialized = true;
            else {
                applyFn.clearColors();
                if (applyFn.setVertexColors(shader, geoPoint) == MS::kSuccess) colorSetInitialized = true;
            }
        }
        else if (changedIndices.length() > 0) {
            double fraction = (double)changedIndices.length() / (double)N;
            if (fraction > 0.85) {
                applyFn.setVertexColors(shader, geoPoint);
            }
            else {
                ensureLength(s_subsetColors, changedIndices.length());
                // Parallel gather for subset colors
                s_tp_parallel_for(0, changedIndices.length(), diffThreads, [&](int tid, int s, int e) {
                    for (int k = s; k < e; ++k) s_subsetColors[k] = shader[changedIndices[k]];
                    });
                applyFn.setVertexColors(s_subsetColors, changedIndices);
            }
        }
    }

    if (plug == stretchMap || plug == finalShp) data.outputValue(stretchMap).set(MFnDoubleArrayData().create(stretchArray));
    if (plug == compressionMap || plug == finalShp) data.outputValue(compressionMap).set(MFnDoubleArrayData().create(compressionArray));

    data.setClean(plug);
    return MStatus::kSuccess;
}

MStatus tensify::preEvaluation(const MDGContext&, const MEvaluationNode& evaluationNode) {
    if (evaluationNode.dirtyPlugExists(MPlug(thisMObject(), refShp))) refDrt = true;
    if (evaluationNode.dirtyPlugExists(MPlug(thisMObject(), shp))) shpDrt = true;
    if (evaluationNode.dirtyPlugExists(MPlug(thisMObject(), clrAttribute))) rampDirty = true;
    return MStatus::kSuccess;
}

MStatus tensify::setDependentsDirty(const MPlug& drtPlg, MPlugArray&) {
    refDrt = (drtPlg.partialName() == REF_INPUT);
    shpDrt = (drtPlg.partialName() == INPUT);
    if (drtPlg.partialName().indexW("tensifyColor") != -1) rampDirty = true;
    MPlug pClr(thisMObject(), clrAttribute);
    if (!pClr.isNull() && drtPlg == pClr) rampDirty = true;
    return MStatus::kSuccess;
}
