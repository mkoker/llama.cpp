# Running Man Animation

> Build a professional-looking animated vector video of a running man using Manim CE — filled shapes, proper body proportions, smooth limb mechanics, no stick figures.

### PHASE 1 · Environment Setup

- [x] T01 Install and verify Manim CE + ffmpeg
  GATES:
    build: manim --version 2>&1 | grep -q 'Manim Community'
    test:  manim --version 2>&1 && ffmpeg -version 2>&1 | grep -q 'ffmpeg version'

- [ ] T02 Create project directory structure
  GATES:
    build: ssh ubuntu@192.168.1.200 'test -d /home/ubuntu/projects/running-man-animation && test -f /home/ubuntu/projects/running-man-animation/plan.md'
    test:  ssh ubuntu@192.168.1.200 'cd /home/ubuntu/projects/running-man-animation && ls -la script.py plan.md 2>/dev/null | wc -l | grep -q 2'

### PHASE 2 · Character Design

- [ ] T03 Implement RunningMan class with proper proportions
  GATES:
    build: ssh ubuntu@192.168.1.200 'cd /home/ubuntu/projects/running-man-animation && grep -c "class RunningMan" script.py'
    test:  ssh ubuntu@192.168.1.200 'cd /home/ubuntu/projects/running-man-animation && python3 -c "import ast; ast.parse(open(\"script.py\").read()); print(\"syntax ok\")"'

- [ ] T04 Add filled shapes for torso, head, limbs (no stick figures)
  GATES:
    build: ssh ubuntu@192.168.1.200 'cd /home/ubuntu/projects/running-man-animation && grep -cE "(Ellipse|Circle|Rectangle|VGroup)" script.py'
    test:  ssh ubuntu@192.168.1.200 'cd /home/ubuntu/projects/running-man-animation && python3 -c "import ast; tree=ast.parse(open(\"script.py\").read()); print(\"parse ok\")"'

### PHASE 3 · Animation Mechanics

- [ ] T05 Implement running cycle with synchronized arm/leg swings
  GATES:
    build: ssh ubuntu@192.168.1.200 'cd /home/ubuntu/projects/running-man-animation && grep -c "ValueTracker\|add_updater\|always_redraw\|rot\|set_angled" script.py'
    test:  ssh ubuntu@192.168.1.200 'cd /home/ubuntu/projects/running-man-animation && python3 -c "import ast; ast.parse(open(\"script.py\").read()); print(\"parse ok\")"'

- [ ] T06 Add ground line, motion trail, and background styling
  GATES:
    build: ssh ubuntu@192.168.1.200 'cd /home/ubuntu/projects/running-man-animation && grep -cE "(Line.*DOWN|Line.*RIGHT|background_color|set_fill)" script.py'
    test:  ssh ubuntu@192.168.1.200 'cd /home/ubuntu/projects/running-man-animation && python3 -c "import ast; ast.parse(open(\"script.py\").read()); print(\"parse ok\")"'

### PHASE 4 · Render & Output

- [ ] T07 Render draft at 480p, fix any issues
  GATES:
    build: ssh ubuntu@192.168.1.200 'cd /home/ubuntu/projects/running-man-animation && manim -ql script.py RunningScene 2>&1 | tail -1 | grep -qE "(Success|ERROR)" && test -f media/videos/script/480p15/RunningScene.mp4'
    test:  ssh ubuntu@192.168.1.200 'cd /home/ubuntu/projects/running-man-animation && ffprobe -v error -show_entries stream=codec_name,duration -of csv=p=0 media/videos/script/480p15/RunningScene.mp4 | grep -q "h264"'

- [ ] T08 Render final production version (720p or 1080p)
  GATES:
    build: ssh ubuntu@192.168.1.200 'cd /home/ubuntu/projects/running-man-animation && manim -qm script.py RunningScene 2>&1 | tail -1 | grep -qE "(Success|ERROR)" && test -f media/videos/script/720p30/RunningScene.mp4'
    test:  ssh ubuntu@192.168.1.200 'ffprobe -v error -show_entries stream=width,height,codec_name,duration -of json media/videos/script/720p30/RunningScene.mp4 | python3 -c "import sys,json;d=json.load(sys.stdin);s=d[\"streams\"][0];assert s[\"width\"]>=1280 or s[\"height\"]>=720;assert s[\"codec_name\"]==\"h264\";print(\"ok\")"'
