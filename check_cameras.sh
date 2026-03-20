#!/bin/bash
#
# Script kiểm tra camera USB trên Jetson Nano
#

echo "============================================"
echo " Camera Check - Jetson Nano"
echo "============================================"

echo ""
echo "[1] Video devices detected:"
ls -la /dev/video* 2>/dev/null || echo "  No video devices found!"

echo ""
echo "[2] USB devices:"
lsusb | grep -i "camera\|video\|ov9732\|0c45\|sonix" || echo "  No USB cameras detected via lsusb"

echo ""
echo "[3] V4L2 camera info:"
for dev in /dev/video*; do
    if [ -e "$dev" ]; then
        echo ""
        echo "  --- $dev ---"
        v4l2-ctl --device="$dev" --info 2>/dev/null | head -5
        echo "  Formats:"
        v4l2-ctl --device="$dev" --list-formats-ext 2>/dev/null | head -20
    fi
done

echo ""
echo "[4] Python camera test:"
python3 -c "
import cv2
for i in range(4):
    cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
    if cap.isOpened():
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ret, frame = cap.read()
        status = 'OK' if ret else 'NO FRAME'
        print(f'  /dev/video{i}: {w}x{h} - {status}')
        cap.release()
    else:
        print(f'  /dev/video{i}: Not available')
"

echo ""
echo "============================================"
echo " Nếu không thấy camera, thử:"
echo "  1. Rút và cắm lại USB camera"
echo "  2. sudo apt install v4l-utils"
echo "  3. Kiểm tra USB hub có đủ nguồn"
echo "============================================"
