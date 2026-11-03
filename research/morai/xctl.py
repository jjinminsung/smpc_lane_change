#!/usr/bin/env python3
"""X11 조작 도구 (ctypes + libXtst). xdotool 대신.
  xctl.py activate <winid>            창을 맨 앞으로, 포커스
  xctl.py geom <winid>                창의 화면 절대 좌표와 크기
  xctl.py wclick <winid> <rx> <ry> [button] [double]   창 기준 좌표 클릭
  xctl.py wmove <winid> <rx> <ry>     창 기준 좌표로 마우스 이동
  xctl.py key <keysym> [mod...]       키 (예: F4, Return, Escape, Tab / mod: Control_L Shift_L Alt_L)
  xctl.py type <text>                 ASCII 문자열 입력
  xctl.py shot <winid> <out.png> [scale]  창 캡처 (import)
"""
import ctypes
import ctypes.util
import os
import subprocess
import sys
import time

DISPLAY = os.environ.get('DISPLAY', ':1').encode()
x11 = ctypes.cdll.LoadLibrary('libX11.so.6')
xtst = ctypes.cdll.LoadLibrary('libXtst.so.6')

x11.XOpenDisplay.restype = ctypes.c_void_p
x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
x11.XDefaultRootWindow.restype = ctypes.c_ulong
x11.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
x11.XInternAtom.restype = ctypes.c_ulong
x11.XInternAtom.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
x11.XRaiseWindow.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
x11.XSetInputFocus.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
x11.XFlush.argtypes = [ctypes.c_void_p]
x11.XMoveResizeWindow.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_int, ctypes.c_uint, ctypes.c_uint]
x11.XSync.argtypes = [ctypes.c_void_p, ctypes.c_int]
x11.XStringToKeysym.restype = ctypes.c_ulong
x11.XStringToKeysym.argtypes = [ctypes.c_char_p]
x11.XKeysymToKeycode.restype = ctypes.c_ubyte
x11.XKeysymToKeycode.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
x11.XTranslateCoordinates.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_int, ctypes.c_int,
                                      ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
                                      ctypes.POINTER(ctypes.c_ulong)]
xtst.XTestFakeMotionEvent.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_ulong]
xtst.XTestFakeButtonEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_ulong]
xtst.XTestFakeKeyEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_ulong]


class XWindowAttributes(ctypes.Structure):
    _fields_ = [('x', ctypes.c_int), ('y', ctypes.c_int), ('width', ctypes.c_int), ('height', ctypes.c_int),
                ('border_width', ctypes.c_int), ('depth', ctypes.c_int), ('visual', ctypes.c_void_p),
                ('root', ctypes.c_ulong), ('pad', ctypes.c_byte * 128)]


x11.XGetWindowAttributes.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(XWindowAttributes)]


class XClientMessageEvent(ctypes.Structure):
    _fields_ = [('type', ctypes.c_int), ('serial', ctypes.c_ulong), ('send_event', ctypes.c_int),
                ('display', ctypes.c_void_p), ('window', ctypes.c_ulong), ('message_type', ctypes.c_ulong),
                ('format', ctypes.c_int), ('l', ctypes.c_long * 5)]


class XEvent(ctypes.Union):
    _fields_ = [('xclient', XClientMessageEvent), ('pad', ctypes.c_long * 24)]


x11.XSendEvent.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_long, ctypes.POINTER(XEvent)]

d = x11.XOpenDisplay(DISPLAY)
if not d:
    sys.exit('cannot open display %s' % DISPLAY.decode())
root = x11.XDefaultRootWindow(d)


def geom(win):
    x, y, child = ctypes.c_int(), ctypes.c_int(), ctypes.c_ulong()
    x11.XTranslateCoordinates(d, win, root, 0, 0, ctypes.byref(x), ctypes.byref(y), ctypes.byref(child))
    a = XWindowAttributes()
    x11.XGetWindowAttributes(d, win, ctypes.byref(a))
    return x.value, y.value, a.width, a.height


def activate(win):
    ev = XEvent()
    ev.xclient.type = 33  # ClientMessage
    ev.xclient.send_event = 1
    ev.xclient.display = d
    ev.xclient.window = win
    ev.xclient.message_type = x11.XInternAtom(d, b'_NET_ACTIVE_WINDOW', 0)
    ev.xclient.format = 32
    ev.xclient.l[0] = 2  # 페이저/사용자 요청
    ev.xclient.l[1] = 0
    ev.xclient.l[2] = 0
    x11.XSendEvent(d, root, 0, (1 << 20) | (1 << 19), ctypes.byref(ev))
    x11.XRaiseWindow(d, win)
    x11.XSync(d, 0)
    time.sleep(0.3)
    x11.XSetInputFocus(d, win, 2, 0)
    x11.XSync(d, 0)


def move(x, y):
    xtst.XTestFakeMotionEvent(d, -1, int(x), int(y), 0)
    x11.XSync(d, 0)


def click(x, y, button=1, double=False):
    move(x, y)
    time.sleep(0.15)
    for _ in range(2 if double else 1):
        xtst.XTestFakeButtonEvent(d, button, 1, 0)
        x11.XSync(d, 0)
        time.sleep(0.05)
        xtst.XTestFakeButtonEvent(d, button, 0, 0)
        x11.XSync(d, 0)
        time.sleep(0.08)


def key(name, mods=()):
    codes = [x11.XKeysymToKeycode(d, x11.XStringToKeysym(m.encode())) for m in mods]
    kc = x11.XKeysymToKeycode(d, x11.XStringToKeysym(name.encode()))
    if not kc:
        sys.exit('unknown keysym %s' % name)
    for c in codes:
        xtst.XTestFakeKeyEvent(d, c, 1, 0)
    xtst.XTestFakeKeyEvent(d, kc, 1, 0)
    x11.XSync(d, 0)
    time.sleep(0.05)
    xtst.XTestFakeKeyEvent(d, kc, 0, 0)
    for c in reversed(codes):
        xtst.XTestFakeKeyEvent(d, c, 0, 0)
    x11.XSync(d, 0)


def type_text(s):
    for ch in s:
        name = {' ': 'space', '.': 'period', '_': 'underscore', '-': 'minus', '/': 'slash'}.get(ch, ch)
        if ch.isupper() or ch == '_':
            key(name if ch == '_' else ch, ('Shift_L',) if ch.isupper() else ('Shift_L',))
        else:
            key(name)
        time.sleep(0.03)


def main():
    cmd = sys.argv[1]
    a = sys.argv[2:]
    if cmd == 'activate':
        activate(int(a[0], 0))
    elif cmd == 'resize':
        x11.XMoveResizeWindow(d, int(a[0], 0), int(a[1]), int(a[2]), int(a[3]), int(a[4]))
        x11.XSync(d, 0)
    elif cmd == 'geom':
        print('%d %d %d %d' % geom(int(a[0], 0)))
    elif cmd in ('wclick', 'wmove'):
        x0, y0, _, _ = geom(int(a[0], 0))
        x, y = x0 + int(a[1]), y0 + int(a[2])
        if cmd == 'wmove':
            move(x, y)
        else:
            click(x, y, int(a[3]) if len(a) > 3 else 1, len(a) > 4 and a[4] == 'double')
        print('at %d %d' % (x, y))
    elif cmd == 'key':
        key(a[0], tuple(a[1:]))
    elif cmd == 'type':
        type_text(a[0])
    elif cmd == 'shot':
        out = a[1]
        subprocess.run(['import', '-window', a[0], out], check=True, env=dict(os.environ, DISPLAY=DISPLAY.decode()))
        if len(a) > 2:
            from PIL import Image
            im = Image.open(out)
            sc = float(a[2])
            im.resize((int(im.size[0] * sc), int(im.size[1] * sc))).save(out)
    else:
        sys.exit(__doc__)


if __name__ == '__main__':
    main()
