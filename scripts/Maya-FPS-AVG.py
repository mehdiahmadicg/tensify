# Run in Script Editor -> Python
# ---------------------------------------------------------------------------
#  Maya viewport playback benchmark -- final results in light green
# ---------------------------------------------------------------------------
import time
import csv

from maya import cmds

try:
    timer = time.perf_counter          # Python 3 (Maya 2022+)
except AttributeError:
    timer = time.clock                 # Python 2 (Maya <= 2020)

try:
    from PySide6 import QtWidgets, QtGui, QtCore      # Maya 2025+
    import shiboken6 as shiboken
except ImportError:
    from PySide2 import QtWidgets, QtGui, QtCore      # Maya 2017 - 2024
    import shiboken2 as shiboken

RESULT_COLOR         = "#90EE90"   # light green; any "#RRGGBB" works
REPORTER_NAME_HINT   = "cmdScrollFieldReporter"   # class/object-name substring
POPUP_ON_FALLBACK    = True        # show green popup if SE can't be coloured

# ---------------------------------------------------------------------------
#  Coloured printing into the Script Editor history pane (via Qt)
# ---------------------------------------------------------------------------
_reporter        = None
_lookup_failed   = False
_fallback_lines  = []


def _get_reporter():
    """Scan all Qt widgets for the Script Editor history pane and return it
    wrapped as its real text-edit class, or None if not found."""
    best = None
    hint = REPORTER_NAME_HINT.lower()
    for w in QtWidgets.QApplication.allWidgets():
        try:
            cls  = str(w.metaObject().className() or "")
            name = str(w.objectName() or "")
        except Exception:
            continue
        if hint in cls.lower() or hint in name.lower():
            if best is None or (w.isVisible() and not best.isVisible()):
                best = w                      # prefer a visible editor
    if best is None:
        return None
    # Resolve the concrete base class before re-wrapping the C++ pointer
    meta, classes = best.metaObject(), set()
    while meta is not None:
        classes.add(meta.className())
        meta = meta.superClass()
    ptr = shiboken.getCppPointer(best)[0]
    if "QPlainTextEdit" in classes:
        return shiboken.wrapInstance(ptr, QtWidgets.QPlainTextEdit)
    if "QTextEdit" in classes:
        return shiboken.wrapInstance(ptr, QtWidgets.QTextEdit)
    return None


def se_print(text="", color=RESULT_COLOR):
    """Print to the Script Editor in 'color'. Falls back to plain print()
    (with a loud warning and optionally a green popup) if the history
    pane cannot be found or written to."""
    global _reporter, _lookup_failed
    if color is None:
        print(text)
        return
    if _reporter is None and not _lookup_failed:
        _reporter = _get_reporter()
        if _reporter is None:
            _lookup_failed = True
            print("[se_print] WARNING: Script Editor history widget not found - "
                  "printing without colour. Run debug_script_editor() to see "
                  "what your Maya build provides and adjust REPORTER_NAME_HINT.")
    if _reporter is None:
        _fallback_lines.append(text)
        print(text)
        return
    try:
        cursor = _reporter.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        fmt = QtGui.QTextCharFormat()
        fmt.setForeground(QtGui.QColor(color))
        cursor.insertText(text + "\n", fmt)
        _reporter.setTextCursor(cursor)     # keep the pane scrolled down
        _reporter.ensureCursorVisible()
    except Exception:
        _reporter = None                    # don't retry a broken widget
        _fallback_lines.append(text)
        print(text)


def debug_script_editor():
    """Diagnostic: list every Qt widget whose class/object name looks like a
    script-editor history pane. Run this if results come out uncoloured."""
    hits = 0
    for w in QtWidgets.QApplication.allWidgets():
        try:
            cls, name = w.metaObject().className(), w.objectName()
        except Exception:
            continue
        if any(k in cls.lower() + name.lower()
               for k in ("reporter", "scripteditor", "script editor")):
            print("class={0:<30} objectName={1:<35} visible={2}"
                  .format(cls, name, w.isVisible()))
            hits += 1
    if not hits:
        print("No candidate history widgets found.")


def show_results_popup(lines, rgb=RESULT_COLOR):
    """Safety net: exact-colour results in a small always-on-top window."""
    global _popup_ref
    _popup_ref = QtWidgets.QLabel(
        "<pre style='color:{0}; font-size:12px'>{1}</pre>"
        .format(rgb, "<br>".join(lines)))
    _popup_ref.setWindowTitle("Playback benchmark")
    _popup_ref.setWindowFlags(_popup_ref.windowFlags()
                              | QtCore.Qt.WindowStaysOnTopHint)
    _popup_ref.show()


# ---------------------------------------------------------------------------
#  Playback benchmark (Maya's real playback engine -- unchanged logic)
# ---------------------------------------------------------------------------
def benchmark_playback(runs=1, warmup=None, playback_speed=0, csv_path=None):
    """
    runs           : number of timed playback passes (results are averaged)
    warmup         : one untimed pass first so shaders/textures/caches are hot
    playback_speed : 0 -> free (play every frame), 1 -> real-time,
                     2 -> half, 3 -> twice, None -> keep the user's setting
    csv_path       : optional file path; per-run results are written as CSV
    """
    global _fallback_lines

    saved_loop  = cmds.playbackOptions(query=True, loop=True)
    saved_speed = cmds.playbackOptions(query=True, playbackSpeed=True)

    cmds.playbackOptions(loop="once")
    if playback_speed is not None:
        cmds.playbackOptions(playbackSpeed=playback_speed)

    start_frame = int(cmds.playbackOptions(query=True, min=True))
    end_frame   = int(cmds.playbackOptions(query=True, max=True))
    steps = end_frame - start_frame
    if steps < 1:
        cmds.playbackOptions(loop=saved_loop)
        raise RuntimeError("Playback range must span more than one frame "
                           "(min={0}, max={1}).".format(start_frame, end_frame))

    def play_pass():
        cmds.currentTime(start_frame, edit=True)
        t0 = timer()
        cmds.play(wait=True)
        t1 = timer()
        return t1 - t0

    results = []
    try:
        print("Scene fps setting : {0}".format(cmds.currentUnit(query=True, time=True)))
        print("Playback range    : {0} - {1} ({2} frame steps/pass)"
              .format(start_frame, end_frame, steps))
        print("Speed mode        : {0}"
              .format({None: "user setting",
                       0: "free (play every frame)",
                       1: "real-time",
                       2: "half",
                       3: "twice"}.get(playback_speed, playback_speed)))

        if warmup:
            print("Warm-up pass (not timed)...")
            play_pass()

        se_print("-" * 44)
        for run in range(1, runs + 1):
            elapsed = play_pass()
            fps = steps / elapsed if elapsed > 0 else 0.0
            results.append((run, elapsed, fps))            
        se_print("-" * 44)
        #se_print("Run {0}: {1:8.3f} s   ->  {2:7.2f} FPS".format(run, elapsed, fps))
        se_print("-" * 44)
        se_print("Time : {1:8.3f} s ".format(run, elapsed, fps))
        se_print("-" * 44)
        avg_fps  = sum(r[2] for r in results) / len(results)
        best_fps = max(r[2] for r in results)
        se_print("Average FPS : {0:.2f}".format(avg_fps))
       # se_print("Best    FPS : {0:.2f}".format(best_fps))
    finally:
        cmds.playbackOptions(loop=saved_loop)
        cmds.playbackOptions(playbackSpeed=saved_speed)
        cmds.currentTime(start_frame, edit=True)

    if _fallback_lines and POPUP_ON_FALLBACK:
        show_results_popup(_fallback_lines)     # green anyway
        _fallback_lines = []

    if csv_path:
        try:
            fh = open(csv_path, "w", newline="")   # Python 3
        except TypeError:
            fh = open(csv_path, "wb")              # Python 2
        with fh:
            writer = csv.writer(fh)
            writer.writerow(["run", "elapsed_seconds", "fps"])
            writer.writerows(results)
        print("CSV written to    : {0}".format(csv_path))

    return avg_fps


# --------------------------------------------------------------------------
# Watch the viewport while it runs: the animation plays at exactly the
# speed being measured. Compare with HUD -> Frame Rate: they now agree.
# --------------------------------------------------------------------------
benchmark_playback(runs=1, playback_speed=0)

# Diagnostics if the green still doesn't appear:
# se_print("green test")        # instant one-line test
# debug_script_editor()         # lists candidate widgets (class + objectName)