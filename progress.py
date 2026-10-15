import sys

import config


def show_progress(current, total, prefix='进度'):
    """打印单行控制台进度条，不依赖第三方库。"""
    if not config.SHOW_PROGRESS or total <= 0:
        return

    current = min(current, total)
    width = config.PROGRESS_BAR_WIDTH
    filled = int(width * current / total)
    bar = '#' * filled + '-' * (width - filled)
    percent = current / total * 100
    line = f'{prefix}: [{bar}] {current}/{total} {percent:6.2f}%'

    # IDE 运行窗口经常不实时渲染 \r，这里在非 TTY 环境改成按间隔打印新行。
    if not sys.stdout.isatty():
        every = max(int(config.PROGRESS_PRINT_EVERY), 1)

        if current == 0 or current == 1 or current == total or current % every == 0:
            print(line)

        return

    sys.stdout.write('\r' + line)
    sys.stdout.flush()

    if current >= total:
        sys.stdout.write('\n')
        sys.stdout.flush()
