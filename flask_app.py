from __future__ import annotations

from pathlib import Path
from uuid import uuid4
from multiprocessing import get_context
from io import BytesIO
import base64
import os

from flask import Flask, jsonify, render_template_string, request, send_from_directory
from werkzeug.utils import secure_filename
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from running_analyzer import analyze_video, create_annotated_video


ROOT = Path(__file__).resolve().parent
UPLOADS = ROOT / "web_uploads"
UPLOADS.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {"mp4", "mov", "avi", "m4v"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 300 * 1024 * 1024


def landing_knee_chart(landing_df):
    right = landing_df.loc[landing_df["foot"] == "right", "landing_knee_flexion"].mean()
    left = landing_df.loc[landing_df["foot"] == "left", "landing_knee_flexion"].mean()
    if landing_df.empty or not (right == right and left == left):
        return None
    fig, axis = plt.subplots(figsize=(6.6, 3.6))
    bars = axis.bar(["Right landing", "Left landing"], [right, left], color=["#2563eb", "#7c3aed"], width=.52)
    axis.axhspan(15, 25, color="#dcfce7", alpha=.85, label="Target range (15–25°)")
    axis.set_ylabel("Knee flexion (degrees)")
    axis.set_ylim(0, max(35, right, left) + 8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=.18)
    axis.legend(frameon=False, loc="upper right")
    for bar, value in zip(bars, [right, left]):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.1f}°", ha="center", fontweight="bold")
    fig.tight_layout()
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=160, transparent=True)
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _analysis_worker(video_path, height, send_pipe):
    """MediaPipe를 웹 요청과 분리된 프로세스의 메인 스레드에서 실행합니다."""
    try:
        send_pipe.send((True, analyze_video(video_path, height)))
    except BaseException as error:
        send_pipe.send((False, str(error)))
    finally:
        send_pipe.close()


def analyze_in_worker(video_path, height):
    context = get_context("spawn")
    receive_pipe, send_pipe = context.Pipe(duplex=False)
    process = context.Process(target=_analysis_worker, args=(str(video_path), height, send_pipe))
    process.start()
    send_pipe.close()
    if not receive_pipe.poll(240):
        process.terminate()
        process.join()
        raise RuntimeError("분석 시간이 너무 오래 걸렸습니다. 15초 이하의 짧은 영상을 사용해 주세요.")
    ok, payload = receive_pipe.recv()
    process.join()
    if not ok:
        raise RuntimeError(payload)
    return payload


BASE = """<!doctype html><html lang='ko'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>RunForm AI</title><style>
.status-guide{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 16px}.status-guide span{padding:8px 10px;border-radius:8px;font-size:13px;font-weight:600}.status-normal{color:#15803d;background:#dcfce7}.status-warning{color:#dc2626;background:#fee2e2}.status-observe{color:#c2410c;background:#ffedd5}.status-check{color:#6d28d9;background:#ede9fe}.status-reference{color:#1d4ed8;background:#dbeafe}
.video-grid{grid-template-columns:1fr!important}.video-grid div:first-child{display:none}
.splash{position:fixed;inset:0;z-index:20;display:flex;align-items:center;justify-content:center;background:#f4f7fb;transition:opacity .35s ease,visibility .35s ease}.splash.hide{opacity:0;visibility:hidden}.splash-logo{text-align:center;color:#18212f}.splash-logo .emoji{font-size:56px}.splash-logo h1{font-size:42px;margin:12px 0 7px}.splash-logo p{margin:0;color:#657084;font-weight:600}
.drop #video{margin-left:46px;max-width:calc(100% - 46px)}
*{box-sizing:border-box} body{margin:0;background:#f4f7fb;color:#18212f;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}.wrap{max-width:1100px;margin:auto;padding:42px 24px}.hero{margin-bottom:28px}.hero h1{font-size:34px;margin:0 0 8px}.hero p{color:#657084}.card{background:white;border-radius:18px;padding:28px;box-shadow:0 8px 32px #18212f13;margin:18px 0}.grid{display:grid;grid-template-columns:1.3fr .7fr;gap:20px}.drop{position:relative;border:2px dashed #a9bad4;border-radius:12px;padding:24px;background:#fbfcff}.file-name{font-size:14px;color:#46617f;margin-top:10px}.file-clear{display:none;position:absolute;top:10px;left:10px;width:30px;height:30px;padding:0;border-radius:50%;background:#e8eef8;color:#27415f;font-size:21px;line-height:1}.button,button{border:0;border-radius:10px;padding:13px 18px;font-weight:700;cursor:pointer;background:#2563eb;color:white;font-size:15px}.button:disabled,button:disabled{background:#b9c5d6;cursor:not-allowed}.button.secondary,button.secondary{background:#e8eef8;color:#27415f;text-decoration:none;display:inline-block}.hint{color:#6c7788;font-size:14px;line-height:1.6}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:#f5f7fb;border-radius:12px;padding:16px}.metric span{font-size:13px;color:#657084}.metric b{display:block;font-size:22px;margin-top:6px}.feedback{width:100%;border-collapse:collapse}.feedback th,.feedback td{padding:13px;border-bottom:1px solid #e7ebf2;text-align:left;vertical-align:top}.badge{font-weight:700;white-space:nowrap}.video-grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}.video-grid video{width:100%;border-radius:12px;background:#111}.welcome{max-width:780px;margin:44px auto}.welcome h2{font-size:29px;margin:0 0 10px}.guide{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:24px 0}.guide-item{padding:18px;border-radius:13px;background:#f6f8fc;line-height:1.5}.guide-item b{display:block;margin-bottom:5px}.confirm{display:flex;gap:10px;align-items:center;padding:16px;background:#edf4ff;border-radius:12px;margin:22px 0;font-weight:600}.loading{display:none;position:fixed;inset:0;background:#18212fc7;z-index:10;color:white;align-items:center;justify-content:center;text-align:center}.loading.show{display:flex}.spinner{width:48px;height:48px;border:5px solid #ffffff55;border-top-color:#fff;border-radius:50%;margin:0 auto 18px;animation:spin 1s linear infinite}@keyframes spin{to{transform:rotate(360deg)}}@media(max-width:720px){.grid,.video-grid,.guide{grid-template-columns:1fr}.metrics{grid-template-columns:1fr 1fr}}
</style></head><body>{{ splash|safe }}<main class='wrap'><section class='hero'><h1>🏃 RunForm AI</h1><p>러닝 영상을 분석해 자세 피드백을 제공합니다.</p></section>{{ content|safe }}</main></body></html>"""

START = """<section class='card welcome'><h2>달리기 전, 촬영 조건을 확인해 주세요</h2><p class='hint'>좋은 영상일수록 자세 분석 결과가 더 정확해집니다.</p><div class='guide'><div class='guide-item'><b>① 시작 전 1~2초</b>카메라 앞에서 가만히 서 있어 주세요.</div><div class='guide-item'><b>② 카메라는 고정</b>전신과 발이 모두 화면 안에 나오게 촬영하세요.</div><div class='guide-item'><b>③ 가로 방향으로 달리기</b>가능하면 옆모습에서 한 방향으로 달려 주세요.</div><div class='guide-item'><b>④ 7~15초 영상</b>대각선·정면·왕복 영상은 거리 분석 정확도가 낮아질 수 있습니다.</div><div class='guide-item'><b>⑤ 한 사람만 화면에 나오게</b>다른 사람이 함께 나오면 관절 추정 대상이 섞일 수 있습니다.</div></div><label class='confirm'><input id='confirm' type='checkbox'> 촬영 조건을 확인했어요.</label><button id='start' type='button' disabled onclick="location.href='/upload'">영상 분석하러 가기 →</button></section><script>const confirmBox=document.getElementById('confirm'), startButton=document.getElementById('start');confirmBox.addEventListener('change',()=>startButton.disabled=!confirmBox.checked);</script>"""

UPLOAD = """<div id='loading' class='loading'><div><div class='spinner'></div><h2>러닝 자세를 분석하고 있습니다</h2><p>관절 좌표와 움직임 지표를 계산 중입니다. 영상 길이에 따라 잠시 걸릴 수 있어요.</p></div></div><a class='button secondary' href='/'>← 촬영 가이드 보기</a><section class='card'><form id='analysis-form' method='post' action='/analyze' enctype='multipart/form-data'><div class='grid'><div><h2>러닝 영상 업로드</h2><div class='drop'><button id='clear-file' class='file-clear' type='button' aria-label='업로드 파일 제거' onclick='clearFile()'>×</button><input id='video' name='video' type='file' accept='.mp4,.mov,.avi,.m4v' required><div id='file-name' class='file-name'>선택된 파일이 없습니다.</div></div><p class='hint'>권장: 7~15초, 전신·발이 모두 보이는 고정 측면 영상</p></div><div><h2>사용자 정보</h2><label>키 (cm)</label><input name='height' type='number' min='120' max='230' value='170' required style='display:block;width:100%;margin:10px 0 20px;padding:13px;border:1px solid #ccd5e3;border-radius:10px;font-size:17px'><button type='submit'>자세 분석 시작</button></div></div></form></section><script>const input=document.getElementById('video'),nameBox=document.getElementById('file-name'),clearButton=document.getElementById('clear-file');function updateFileState(){const selected=input.files.length>0;nameBox.textContent=selected?input.files[0].name:'선택된 파일이 없습니다.';clearButton.style.display=selected?'block':'none';}input.addEventListener('change',updateFileState);function clearFile(){input.value='';updateFileState();}document.getElementById('analysis-form').addEventListener('submit',()=>document.getElementById('loading').classList.add('show'));</script>"""

FOOT_STRIKE_CARD = """<style>
.feedback th:first-child,.feedback td:first-child{white-space:nowrap;width:1%}
.strike-title{display:flex;align-items:center;gap:8px;margin:0 0 8px}.info-button{width:24px;height:24px;padding:0;border-radius:50%;background:#e8eef8;color:#27415f;font-size:14px;line-height:24px}.strike-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:18px 0}.strike-card{padding:16px;border:1px solid #e1e8f2;border-radius:12px;background:#fbfcff}.strike-card.active{border:2px solid #2563eb;background:#eff6ff}.strike-card span{display:block;color:#657084;font-size:13px}.strike-card b{display:block;margin-top:5px;font-size:19px}.strike-summary{padding:14px 16px;border-radius:12px;background:#f5f7fb;line-height:1.65}.strike-modal{display:none;position:fixed;inset:0;z-index:30;padding:24px;background:#18212f99;align-items:center;justify-content:center}.strike-modal.open{display:flex}.strike-modal-box{position:relative;width:min(680px,100%);max-height:85vh;overflow:auto;padding:28px;border-radius:18px;background:#fff;box-shadow:0 18px 60px #18212f55}.strike-close{position:absolute;top:14px;right:14px;width:32px;height:32px;padding:0;border-radius:50%;background:#e8eef8;color:#27415f;font-size:22px;line-height:1}.strike-modal-box h3{margin:22px 0 6px}.strike-modal-box h3:first-child{margin-top:0}@media(max-width:720px){.strike-grid{grid-template-columns:1fr}}
</style><section class='card'><h2 class='strike-title'>착지 유형 <button class='info-button' type='button' aria-label='착지 유형 설명 열기' onclick="document.getElementById('strike-modal').classList.add('open')">?</button></h2><p class='hint'>착지 후보 프레임에서 발끝과 뒤꿈치의 높이 차이를 비교한 화면 기반 분류입니다.</p>{% if strike.total %}<div class='strike-grid'>{% for key, label in strike.labels.items() %}<div class='strike-card {% if key == strike.dominant %}active{% endif %}'><span>{{ label }}</span><b>{{ strike.counts[key] }}회</b></div>{% endfor %}</div><div class='strike-summary'><b>주된 경향: {{ strike.dominant_label }}</b><br>{% if strike.reliable %}측면에 가까운 고정 촬영 기준으로 계산되었습니다.{% else %}촬영 각도 또는 착지 후보 수 때문에 <b>참고용</b>으로만 해석해 주세요.{% endif %}<br>어느 유형이 무조건 더 좋다는 뜻은 아니며, 통증·속도·훈련 목적과 함께 봐야 합니다.</div>{% else %}<p class='hint'>착지 유형을 분류할 만큼 충분한 착지 후보를 찾지 못했습니다. 전신과 발이 모두 보이는 측면 영상을 사용해 주세요.</p>{% endif %}</section><div id='strike-modal' class='strike-modal' role='dialog' aria-modal='true' aria-label='착지 유형 설명' onclick="if(event.target===this)this.classList.remove('open')"><div class='strike-modal-box'><button class='strike-close' type='button' aria-label='닫기' onclick="document.getElementById('strike-modal').classList.remove('open')">×</button><h3>전족 착지</h3><p>발의 앞부분이 먼저 닿는 경향입니다. 빠른 속도에서는 자연스럽게 나타날 수 있지만, 종아리와 발가락 쪽 부담을 함께 살펴보세요.</p><h3>중족 착지</h3><p>발 중앙부가 비교적 평평하게 닿는 경향입니다. 충격이 발 전체에 분산되는 느낌일 수 있으나, 개인에게 항상 최선이라는 의미는 아닙니다.</p><h3>후족 착지</h3><p>뒤꿈치가 먼저 닿는 경향입니다. 장거리 러닝에서 흔하지만, 보폭과 무릎·정강이의 부담을 함께 점검하세요.</p><p class='hint'>착지 방식을 갑자기 바꾸면 익숙하지 않은 동작 때문에 관절 부담이 커질 수 있습니다. 통증이 있으면 무리한 교정보다 전문가 상담을 우선하세요.</p></div></div>"""

RESULT = """<a class='button secondary' href='/upload'>← 새 영상 분석</a><section class='card'><h2>분석 결과</h2><div class='metrics'><div class='metric'><span>자세 인식률</span><b>{{ '%.1f'|format(summary.detection_rate) }}%</b></div><div class='metric'><span>분석 시간</span><b>{{ '%.1f'|format(summary.duration_sec) }}초</b></div><div class='metric'><span>촬영 신뢰도</span><b>{{ confidence }}</b></div><div class='metric'><span>원근 변화</span><b>{{ '%.1f'|format(summary.scale_change_pct) }}%</b></div></div></section><section class='card'><h2>자세 피드백</h2><div class='status-guide'><span class='status-normal'>정상: 목표 범위 안에 있습니다.</span><span class='status-warning'>주의: 자세를 교정하면 좋습니다.</span><span class='status-observe'>관찰: 경향을 계속 확인하세요.</span><span class='status-check'>확인 필요: 촬영 조건을 점검하세요.</span><span class='status-reference'>참고용: 수치는 참고만 하세요.</span></div><table class='feedback'><thead><tr><th>분석 요소</th><th>판정</th><th>피드백</th></tr></thead><tbody>{% for item in feedback %}<tr><td>{{ item['분석 요소'] }}</td><td class='badge {{ item['status_class'] }}'>{{ item['판정'] }}</td><td>{{ item['피드백'] }}</td></tr>{% endfor %}</tbody></table></section><section class='card'>{% if landing_chart %}<h2>좌우 착지 무릎 비교</h2><p class='hint'>각 발이 착지하는 순간의 평균 무릎 굽힘각입니다. 초록 영역은 목표 범위 15~25도입니다.</p><img style='display:block;width:100%;max-width:680px;margin:auto' src='data:image/png;base64,{{ landing_chart }}' alt='좌우 착지 무릎 굽힘각 비교 그래프'>{% endif %}</section><section class='card'><h2>관절 추정 영상</h2><video controls style='display:block;width:100%;max-width:560px;margin:0 auto;border-radius:12px' src='/files/{{ annotated }}'></video><p><a class='button' href='/files/{{ annotated }}' download>분석 영상 저장</a></p></section>"""

RESULT = RESULT.replace(
    "<span class='status-normal'>정상: 목표 범위 안에 있습니다.</span><span class='status-warning'>주의: 자세를 교정하면 좋습니다.</span><span class='status-observe'>관찰: 경향을 계속 확인하세요.</span>",
    "<span class='status-normal'>정상: 목표 범위 안에 있습니다.</span><span class='status-observe'>관찰: 경향을 계속 확인하세요.</span><span class='status-warning'>주의: 자세를 교정하면 좋습니다.</span>",
)


def page(content, show_splash=False, **context):
    splash = """<div id='splash' class='splash'><div class='splash-logo'><div class='emoji'>🏃</div><h1>RunForm AI</h1><p>Run smarter, run safer.</p></div></div><script>window.setTimeout(()=>document.getElementById('splash').classList.add('hide'),1300);</script>""" if show_splash else ""
    return render_template_string(BASE, splash=splash, content=render_template_string(content, **context))


@app.get("/")
def home():
    return page(START, show_splash=True)


@app.get("/upload")
def upload():
    return page(UPLOAD)


@app.get("/health")
def health():
    """Docker healthcheck and deployment readiness endpoint."""
    return jsonify(status="ok", service="RunForm AI")


@app.post("/analyze")
def analyze():
    file = request.files.get("video")
    if not file or not file.filename:
        return page("<section class='card'><h2>영상 파일을 선택해 주세요.</h2><a class='button' href='/'>돌아가기</a></section>"), 400
    extension = Path(secure_filename(file.filename)).suffix.lower().lstrip(".")
    if extension not in ALLOWED_EXTENSIONS:
        return page("<section class='card'><h2>지원하지 않는 파일 형식입니다.</h2><p>mp4, mov, avi, m4v 파일을 사용해 주세요.</p><a class='button' href='/'>돌아가기</a></section>"), 400
    try:
        height = int(request.form.get("height", 170))
        if not 120 <= height <= 230:
            raise ValueError("키는 120~230cm 사이로 입력해 주세요.")
        token = uuid4().hex
        original = f"{token}.{extension}"
        original_path = UPLOADS / original
        file.save(original_path)
        result = analyze_in_worker(original_path, height)
        annotated = f"{token}_analysis.mp4"
        create_annotated_video(original_path, result["predictions"], UPLOADS / annotated)
        confidence = {"high": "높음", "medium": "보통", "low": "낮음", "unknown": "판별 불가"}[result["summary"]["motion_reliability"]]
        status_classes = {"정상": "status-normal", "주의": "status-warning", "관찰": "status-observe", "확인 필요": "status-check", "참고용": "status-reference"}
        feedback = result["feedback"].to_dict("records")
        for item in feedback:
            item["status_class"] = status_classes.get(item["판정"], "status-observe")
        chart = landing_knee_chart(result["landings"])
        labels = {"forefoot": "전족 착지", "midfoot": "중족 착지", "rearfoot": "후족 착지"}
        summary = result["summary"]
        strike = {
            "labels": labels,
            "counts": summary["foot_strike_counts"],
            "total": summary["foot_strike_total"],
            "dominant": summary["dominant_foot_strike"],
            "dominant_label": labels.get(summary["dominant_foot_strike"], "판별 불가"),
            "reliable": summary["foot_strike_reliable"],
        }
        strike_card = render_template_string(FOOT_STRIKE_CARD, strike=strike)
        result_content = RESULT.replace("<section class='card'>{% if landing_chart %}", strike_card + "<section class='card'>{% if landing_chart %}")
        return page(result_content, summary=summary, feedback=feedback, confidence=confidence, original=original, annotated=annotated, landing_chart=chart)
    except Exception as error:
        return page("<section class='card'><h2>분석을 완료하지 못했습니다.</h2><p>{{ error }}</p><a class='button' href='/'>새 영상 선택</a></section>", error=str(error)), 400


@app.get("/files/<path:filename>")
def files(filename):
    return send_from_directory(UPLOADS, filename)


@app.errorhandler(413)
def file_too_large(_):
    return page("<section class='card'><h2>영상 용량이 너무 큽니다.</h2><p>300MB 이하의 영상을 사용해 주세요.</p><a class='button' href='/'>돌아가기</a></section>"), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8510)), debug=False, threaded=False)
