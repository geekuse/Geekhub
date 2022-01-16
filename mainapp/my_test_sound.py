import asyncio
import os

import sys
import subprocess, re
import threading

from os import pipe

from mygui import *
from PyQt5 import QtCore, QtGui, QtWidgets


# Переменная-флажок которая определяет произносить ли следующее предложение
# или выйти из цикла чтения предложений с помощью break
# mstop=0
#
# Отдельный поток
def thread(my_func):
    def wrapper(*args, **kwargs):
        my_thread = threading.Thread(target=my_func, args=args, kwargs=kwargs)
        my_thread.start()

    return wrapper


PYTHON_PATH = sys.executable
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# # Функция отдельным потоком в которой производится чтение предложений
# # @thread
# async def processing(t):
#     # Формируем командную строку для RHVoice
#     # fp = '../media/test.mp3'
#     s = 'echo «' + t + '» | RHVoice-test -p anna -o "test43.mp3"'
#     # s = 'echo', '«"+t+"»', '| RHVoice-test -p anna'
#     # Запускаем командную строку
#     # subprocess.call(s, shell=True)
#     # with open('test.mp3', 'w') as f:
#     # args = ['RHVoice-test', "-p", "Anna", "-r", '90', "-t", '75', "-o", '../media/test3.mp3']
#     p = await asyncio.create_subprocess_exec(sys.executable, '-c', s, stdin=asyncio.subprocess.PIPE,
#                                              stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
#
#     try:
#         stdout, stderr = await p.communicate(t.encode('utf-8'))
#     except TimeoutError:
#         p.kill()
#         stdout, stderr = await p.communicate(t.encode('utf-8'))


# class MyWin(QtWidgets.QMainWindow):
#     my_signal = QtCore.pyqtSignal(list, name='my_signal')
#
#     def __init__(self, parent=None):
#         QtWidgets.QWidget.__init__(self, parent)
#         self.ui = Ui_MainWindow()
#         self.ui.setupUi(self)
#         # Определяем функцию Mystop при нажатии на вторую кнопку
#         self.ui.pushButton_2.clicked.connect(self.Mystop)
#         # При нажатии на первую кнопку вызываем во втором потоке функцию чтения текста,
#         # передав в нее текст из текстового поля textEdit
#         self.ui.pushButton.clicked.connect(lambda: processing(self.ui.textEdit.toPlainText()))
#
#     # Функция срабатывающая при нажатии кнопки Стоп
#     # Обнуляет переменную-флажок и убивает процесс речи
#     def Mystop(self):
#         PIPE = subprocess.PIPE
#         subprocess.Popen('pkill -9 RHVoice-test', shell=True, stdin=PIPE, stdout=PIPE, stderr=subprocess.STDOUT)
#         global mstop
#         mstop=0

# if __name__=="__main__":
#     app = QtWidgets.QApplication(sys.argv)
#     myapp = MyWin()
#     myapp.show()
#     sys.exit(app.exec_())


# loop = asyncio.get_event_loop()
# loop.run_until_complete(processing('Да, ещё в офисе, работаю. Сегодня до поздна решил поработать. А ты уже баньки...?'))


# def generator():
#     index = 0
#     while True:
#         index = yield index
#         print(index)
#         index += 1
#
#
# gen = generator()
# next(gen)

#
# def tick():
#     for i in range(10):
#         print('Tick')
#         yield
# list(tick())
#
# import cProfile
# cProfile.run('sum((i * 2 for i in range(10000000) if i % 3 == 0 or i % 5 == 0))')
# cProfile.run('sum([i * 2 for i in range(10000000) if i % 3 == 0 or i % 5 == 0])')

# def emit_lines(pattern=None):
#     lines = []
#     for dir_path, dir_names, file_names in os.walk('GeekHub/'):
#         for file_name in file_names:
#             if file_name.endswith('.py'):
#                 for line in open(os.path.join(dir_path, file_name)):
#                     if pattern in line:
#                         lines.append(line)
#     return lines
#
# print(emit_lines())


# import requests
# import re
# def get_pages(link):
#     links_to_visit = []
#     links_to_visit.append(link)
#     while links_to_visit:
#         current_link = links_to_visit.pop(0)
#         page = requests.get(current_link)
#         for url in re.findall('<a href="([^"]+)">', str(page.content)):
#             if url[0] == '/':
#                 url = current_link + url[1:]
#             pattern = re.compile('https?')
#             if pattern.match(url):
#                 links_to_visit.append(url)
#         yield current_link
# webpage = get_pages('http://yandex.ru')
# for result in webpage:
#     print(result)


# def fib(xterm):
#     x1 = 0
#     x2 = 1
#     count = 0
#
#     if xterm<= 0:
#         print("Укажите целое число.")
#     elif xterm == 1:
#         print('Последовательность Фибоначи до', xterm, ':')
#         print(x1)
#     else:
#         while count < xterm:
#             xth = x1 + x2
#             x1 = x2
#             x2 = xth
#             count += 1
#             yield xth
# f = fib(5)
#
# # print(list(fib(100)))
#
# print(next(f))
# print(next(f))
# print(next(f))
# print(next(f))
# print(next(f))
# print(next(f))


# # Создаем список
# alist = [4, 16, 64, 256]
#
#
# # Вычислим квадратный корень, используя генерацию списка
# out = [a**(1/2) for a in alist]
# print(out)
#
# # Используем выражение генератора, чтобы вычислить квадратный корень
# out = (a**(1/2) for a in alist)
# print(out)
# print(next(out))
# print(next(out))
# print(next(out))
# print(next(out))
# # print(next(out))
#
#
# def list_item():
#     for item in alist:
#         yield item
# gen = list_item()
#
# # iter = 0
# # while iter < len(alist):
# #     print(next(gen))
# #     iter += 1
#
# for item in gen:
#     print(item)

#
# def find_prime(max_count):
#     num = 1
#     while True:
#         if num > 1:
#             for i in range(2, num):
#                 if (num % i) == 0:
#                     break
#             else:
#                 yield num
#         num += 1
#         if max_count < num: break
#
# def find_odd_prime(seq):
#     for num in seq:
#         if (num % 2) != 0:
#             yield num
#
# a_pipe_line = find_odd_prime(find_prime(100))
#
# for ele in a_pipe_line:
#     print(ele)
#
#
# # Очень прикольная штука. С помощью него можно передавать выполнение и ждать результата не останавливая процессор и
# # не перезагружая память, т.е не сохраняя где-то в памяти.
#

# from rhvoice_wrapper import TTS
#
# tts = TTS(threads=1)
# text = 'Очень прикольная штука. С помощью него можно передавать выполнение и ждать результата не останавливая процессор и ' \
#        'не перезагружая память, т.е не сохраняя где-то в памяти.'


# def generator_audio(text, voice='anna', format_='wav', buff=4096, sets=None):
#     with tts.say(text, voice, format_, buff, sets) as gen:
#         for chunk in gen:
#             yield chunk
#
# # generator_audio(text)
# tts.to_file(filename='../media' + '/test12.mp3', text=text, voice='Anna', format_='mp3')
# tts.join()

# def _text():
#     with open('wery_large_book.txt') as fp:
#         text = fp.read(5000)
#         while text:
#             yield text
#             text = fp.read(5000)
#
#
# def generator_audio():
#     with tts.say(text) as gen:
#         for chunk in gen:
#             yield chunk
#
#
# generator_audio()
# tts.join()


# #!/usr/bin/env python3
#
# import multiprocessing
# import threading
# import time
# from rhvoice_wrapper import TTS
#
#
# def benchmarks(tts):
#     # PPS - Phrases Per Second
#     ########################
#     # CPU            # PPS #
#     # i7-8700k       #  82 #
#     # i7-4770k       #  64 #
#     # RaspberryPi 4B #  13 #
#     # OrangePi Prime # 4.4 #
#     # OrangePi Zero  # 3.5 #
#     ########################
#     text = 'Так себе, вызовы сэй будут блокировать выполнение'
#     workers = tuple([_Benchmarks(text, tts.say) for _ in range(tts.thread_count)])
#     yield 'Start...'
#     test_time = 30
#     control = None
#     try:
#         while True:
#             work_time = time.perf_counter()
#             time.sleep(test_time)
#             count = sum([w.count for w in workers])
#             sizes = []
#             for worker in workers:
#                 sizes.extend(worker.sizes)
#             work_time = time.perf_counter() - work_time
#             pps = count / work_time
#             yield 'PPS: {:.4f} (run {:.3f} sec)'.format(pps, work_time)
#             if sizes:
#                 if control is None:
#                     control = sizes[0]
#                 avg = sum(sizes) / len(sizes)
#                 assert control == avg, 'Different sizes: {}'.format(sizes)
#
#     finally:
#         [w.join() for w in workers]
#
#
# class _Benchmarks(threading.Thread):
#     def __init__(self, text, say):
#         super().__init__()
#         self._text = text
#         self._say = say
#         self._count = 0
#         self._sizes = []
#         self._work = True
#         self.start()
#
#     def run(self):
#         while self._work:
#             size = 0
#             with self._say(text=self._text, format_='wav') as fp:
#                 for chunk in fp:
#                     size += len(chunk)
#             self._sizes.append(size)
#             self._count += 1
#
#     @property
#     def count(self):
#         try:
#             return self._count
#         finally:
#             self._count = 0
#
#     @property
#     def sizes(self):
#         try:
#             return self._sizes
#         finally:
#             self._sizes = []
#
#     def join(self, timeout=None):
#         if self._work:
#             self._work = False
#             super().join(timeout)
#
#
# def main():
#     tts = TTS(threads=int(multiprocessing.cpu_count() * 1.5))
#     print('Lib version: {}'.format(tts.lib_version))
#     print('Threads: {}'.format(tts.thread_count))
#     print('Formats: {}'.format(tts.formats))
#     print('Voices: {}'.format(tts.voices))
#     max_ = 5
#     for result in benchmarks(tts):
#         print(result)
#         max_ -= 1
#         if not max_:
#             break
#     tts.join()
#
#
# if __name__ == '__main__':
#     main()

#
#
# #!/usr/bin/env python3
#
# import argparse
# import os
# import subprocess
# import time
#
# from rhvoice_wrapper import TTS
#
#
# class Player:
#     APLAY = ['aplay', '-q', '-']
#
#     def __init__(self, dummy=False):
#         self._dummy = dummy
#         self._popen = None
#         self._write = None
#
#     def play_chunk(self, chunk: bytearray):
#         if self._popen is None:
#             self._init()
#         self._write(chunk)
#
#     def close(self):
#         if self._popen is None:
#             pass
#         elif isinstance(self._popen, subprocess.Popen):
#             self._popen.stdin.close()
#             try:
#                 self._popen.wait(5)
#             except subprocess.TimeoutExpired:
#                 pass
#             self._popen.kill()
#         else:
#             self._popen.close()
#         self._popen = None
#         self._write = None
#
#     def _init(self):
#         if self._dummy:
#             self._popen = open(os.devnull, 'wb')
#             self._write = self._popen.write
#         else:
#             self._popen = subprocess.Popen(self.APLAY, stdin=subprocess.PIPE)
#             self._write = self._popen.stdin.write
#
#
# def pretty_time(sec) -> str:
#     ends = ['sec', 'ms', 'ns']
#     max_index = len(ends) - 1
#     index = 0
#     while sec < 1 and index < max_index and sec:
#         sec *= 1000
#         index += 1
#     result = '{:.2f} {}'.format(sec, ends[index])
#     return '{:>10}'.format(result)
#
#
# def pretty_size(size):
#     ends = ['byte', 'KB', 'MB']
#     index = 0
#     for _ in ends:
#         if size < 1024:
#             break
#         size /= 1024
#         index += 1
#     result = '{:.2f} {}'.format(size, ends[index])
#     return '{:>11}'.format(result)
#
#
# def arg_parser():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-f', '--file', type=open, help='Text file')
#     parser.add_argument('-v', '--voice', default='anna', help='Voice (anna)')
#     parser.add_argument('-q', '--quiet', action='store_true', help='Don\'t play voice (False')
#     parser.add_argument('-c', '--chunks', default=500, type=int, help='Number of characters at a time (500)')
#     return parser.parse_args()
#
#
# def _print(text_size, reply_size, reply_time, full_time):
#     print('Send: {}, Receive: {}, Reply time: {}, Full time: {}'.format(
#         text_size, pretty_size(reply_size), pretty_time(reply_time), pretty_time(full_time)))
#
#
# def main():
#     args = arg_parser()
#     play = Player(dummy=args.quiet)
#     tts = TTS(threads=1)
#     text_size_all = 0
#     reply_size_all = 0
#     reply_time_all = 0
#     full_time_all = 0
#     counts = 0
#     while True:
#         text = args.file.read(args.chunks)
#         if not text:
#             break
#         text_size = len(text)
#         reply_size = 0
#         full_time = time.time()
#         reply_time = None
#         with tts.say(text, voice=args.voice, format_='wav') as say:
#             for chunk in say:
#                 if reply_time is None:
#                     reply_time = time.time() - full_time
#                 reply_size += len(chunk)
#                 play.play_chunk(chunk)
#         full_time = time.time() - full_time
#
#         text_size_all += text_size
#         reply_size_all += reply_size
#         reply_time_all += reply_time
#         full_time_all += full_time
#         counts += 1
#         _print(text_size, reply_size, reply_time, full_time)
#     args.file.close()
#     play.close()
#     tts.join()
#     if counts:
#         print('Summary:')
#         _print(text_size_all, reply_size_all, reply_time_all / counts, full_time_all)
#     print('bye.')
#
#
# if __name__ == '__main__':
#     main()



#!/usr/bin/env python3

import os
import sys
from shlex import quote
from urllib import parse

from flask import Flask, request, make_response, Response, stream_with_context, escape
from rhvoice_wrapper import TTS

from rhvoice_rest_cache import CacheWorker

try:
    from rhvoice_tools.text_prepare import text_prepare
except ImportError as err:
    print('Warning! Preprocessing disable: {}'.format(err))

    def text_prepare(text, stress_marker=False, debug=False):
        return text

DEFAULT_VOICE = 'anna'

FORMATS = {'wav': 'audio/wav', 'mp3': 'audio/mpeg', 'opus': 'audio/ogg', 'flac': 'audio/flac'}
DEFAULT_FORMAT = 'mp3'

app = Flask(__name__, static_url_path='')


def voice_streamer_nocache(text, voice, format_, sets):
    with tts.say(text, voice, format_, None, sets or None) as read:
        for chunk in read:
            yield chunk


def voice_streamer_cache(text, voice, format_, sets):
    inst = cache.get(text, voice, format_, sets)
    try:
        for chunk in inst.read():
            yield chunk
    finally:
        inst.release()


def chunked_stream(stream):
    b_break = b'\r\n'
    for chunk in stream:
        yield format(len(chunk), 'x').encode() + b_break + chunk + b_break
    yield b'0' + b_break * 2


def set_headers():
    if CHUNKED_TRANSFER:
        return {'Transfer-Encoding': 'chunked', 'Connection': 'keep-alive'}


@app.route('/say')
def say():
    text = ' '.join([x for x in parse.unquote(request.args.get('text', '')).splitlines() if x])
    voice = request.args.get('voice', DEFAULT_VOICE)
    format_ = request.args.get('format', DEFAULT_FORMAT)

    if voice not in SUPPORT_VOICES:
        return make_response('Unknown voice: \'{}\'. Support: {}.'.format(escape(voice), ', '.join(SUPPORT_VOICES)), 400)
    if format_ not in FORMATS:
        return make_response('Unknown format: \'{}\'. Support: {}.'.format(escape(format_), ', '.join(FORMATS)), 400)
    if not text:
        return make_response('Unset \'text\'.', 400)

    text = quote(text_prepare(text))
    sets = _get_sets(request.args)
    stream = voice_streamer(text, voice, format_, sets)
    if CHUNKED_TRANSFER:
        stream = chunked_stream(stream)
    return Response(stream_with_context(stream), mimetype=FORMATS[format_], headers=set_headers())


def _normalize_set(val):  # 0..100 -> -1.0..1
    try:
        return max(0, min(100, int(val)))/50.0-1
    except (TypeError, ValueError):
        return 0.0


def _get_sets(args):
    keys = {'rate': 'absolute_rate', 'pitch': 'absolute_pitch', 'volume': 'absolute_volume'}
    return {keys[key]: _normalize_set(args[key]) for key in keys if key in args}


def _get_def(any_, test, def_=None):
    if test not in any_ and len(any_):
        return def_ if def_ and def_ in any_ else next(iter(any_))
    return test


def _check_env(word: str) -> bool:
    return word in os.environ and os.environ[word].lower() not in ['no', 'disable', 'false', '']


def _get_cache_path():
    # Включаем поддержку кэша возвращая путь до него, или None
    if _check_env('RHVOICE_FCACHE'):
        path = os.path.join(os.path.abspath(sys.path[0]), 'rhvoice_rest_cache')
        os.makedirs(path, exist_ok=True)
        return path


def cache_init() -> CacheWorker or None:
    path = _get_cache_path()
    dyn_cache = _check_env('RHVOICE_DYNCACHE')
    return CacheWorker(path, tts.say) if path or dyn_cache else None


if __name__ == "__main__":
    tts = TTS()

    cache = cache_init()
    voice_streamer = voice_streamer_cache if cache else voice_streamer_nocache
    CHUNKED_TRANSFER = _check_env('CHUNKED_TRANSFER')
    print('Chunked transfer encoding: {}'.format(CHUNKED_TRANSFER))

    formats = tts.formats
    DEFAULT_FORMAT = _get_def(formats, DEFAULT_FORMAT, 'wav')
    FORMATS = {key: val for key, val in FORMATS.items() if key in formats}

    SUPPORT_VOICES = tts.voices
    DEFAULT_VOICE = _get_def(SUPPORT_VOICES, DEFAULT_VOICE)
    SUPPORT_VOICES = set(SUPPORT_VOICES)

    print('Threads: {}'.format(tts.thread_count))
    app.run(host='0.0.0.0', port=8080, threaded=True)
    if cache:
        cache.join()
    tts.join()