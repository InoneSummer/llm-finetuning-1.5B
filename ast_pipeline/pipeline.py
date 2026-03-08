"""
HTML 코드 검증 및 품질 평가 파이프라인
LangGraph + AST(parse.js) + Ollama(qwen3.5:35b)

흐름:
  입력 HTML
    → parse_code  : parse.js로 문법 검증
    → fix_code    : 오류 있으면 qwen한테 수정 요청 (최대 3회)
    → score_code  : AST 지표로 품질 점수 산출
    → 출력 결과
"""

import json
import subprocess
import requests
from typing import TypedDict
from langgraph.graph import StateGraph, END


# ── 1. State 정의 ──────────────────────────────────────────────
# 모든 노드가 이 딕셔너리를 공유하며 읽고 씁니다

class HTMLState(TypedDict):
    html: str            # 처리할 HTML 코드
    valid: bool          # 문법 오류 없으면 True
    error: str           # 파싱 에러 메시지 (있을 경우)
    attempts: int        # fix_code 시도 횟수
    components: int      # React 컴포넌트 수
    hooks: int           # 훅 사용 수
    jsx_elements: int    # JSX 엘리먼트 수
    score: float         # 최종 품질 점수


# ── 2. 설정 ────────────────────────────────────────────────────

PARSE_JS_PATH = "./ast_pipeline/parse.js"   # parse.js 경로
MAX_ATTEMPTS = 3                             # 최대 수정 시도 횟수
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3.5:35b"


# ── 3. 노드 함수들 ─────────────────────────────────────────────

def parse_code(state: HTMLState) -> HTMLState:
    """
    [Node 1] parse.js를 호출해서 HTML 문법 검증
    
    - subprocess로 node parse.js 실행
    - 결과 JSON을 파싱해서 state에 저장
    """
    print(f"[parse_code] 검증 중... (시도 횟수: {state['attempts']})")
    
    try:
        result = subprocess.run(
            ["node", PARSE_JS_PATH, state["html"]],
            capture_output=True,
            text=True,
            timeout=10
        )
        parsed = json.loads(result.stdout)
        
        return {
            **state,
            "valid": parsed["valid"],
            "error": parsed.get("error") or "",
            "components": parsed.get("components", 0),
            "hooks": parsed.get("hooks", 0),
            "jsx_elements": parsed.get("jsx_elements", 0),
        }
    
    except Exception as e:
        return {
            **state,
            "valid": False,
            "error": str(e),
        }


def fix_code(state: HTMLState) -> HTMLState:
    """
    [Node 2] Ollama(qwen3.5:35b)에게 HTML 수정 요청
    
    - 에러 메시지와 함께 코드를 보내서 수정된 버전을 받음
    - attempts 카운트 증가
    """
    print(f"[fix_code] 수정 요청 중... (에러: {state['error']})")
    
    prompt = f"""아래 HTML/JSX 코드에 문법 오류가 있습니다.
오류 메시지: {state['error']}

원본 코드:
{state['html']}

오류를 수정한 코드만 출력하세요. 설명 없이 코드만."""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=300
        )
        fixed_html = response.json()["response"].strip()
        
        # 코드 블록 마크다운 제거 (```html ... ``` 형태로 올 경우)
        if fixed_html.startswith("```"):
            lines = fixed_html.split("\n")
            fixed_html = "\n".join(lines[1:-1])
        
        return {
            **state,
            "html": fixed_html,
            "attempts": state["attempts"] + 1,
        }
    
    except Exception as e:
        print(f"[fix_code] Ollama 오류: {e}")
        return {
            **state,
            "attempts": state["attempts"] + 1,
        }


def score_code(state: HTMLState) -> HTMLState:
    """
    [Node 3] AST 지표 기반 품질 점수 산출
    
    점수 공식:
      components   * 10  (컴포넌트 구조화 여부)
      jsx_elements * 2   (UI 풍부도)
      hooks        * 5   (상태 관리 복잡도)
      valid 보너스  + 20  (문법 정확성)
    """
    print("[score_code] 점수 산출 중...")
    
    score = (
        state["components"] * 10 +
        state["jsx_elements"] * 2 +
        state["hooks"] * 5 +
        (20 if state["valid"] else 0)
    )
    
    print(f"[score_code] 점수: {score} "
          f"(components={state['components']}, "
          f"jsx={state['jsx_elements']}, "
          f"hooks={state['hooks']})")
    
    return {**state, "score": float(score)}


# ── 4. 조건부 엣지 함수 ────────────────────────────────────────

def should_fix(state: HTMLState) -> str:
    """
    parse_code 이후 분기 결정:
      - valid=True  → score_code로 이동
      - valid=False + 시도 횟수 초과 → score_code로 이동 (포기)
      - valid=False + 시도 가능    → fix_code로 이동
    """
    if state["valid"]:
        print("[분기] 문법 OK → score로 이동")
        return "score"
    
    if state["attempts"] >= MAX_ATTEMPTS:
        print(f"[분기] {MAX_ATTEMPTS}회 시도 초과 → score로 이동 (포기)")
        return "score"
    
    print(f"[분기] 문법 오류 → fix로 이동 (attempts={state['attempts']})")
    return "fix"


# ── 5. 그래프 조립 ─────────────────────────────────────────────

def build_pipeline() -> StateGraph:
    """
    LangGraph 그래프 생성
    
    노드 추가 → 엣지 연결 → 시작점 설정 → 컴파일
    """
    graph = StateGraph(HTMLState)
    
    # 노드 등록
    graph.add_node("parse_code", parse_code)
    graph.add_node("fix_code", fix_code)
    graph.add_node("score_code", score_code)
    
    # 시작점: parse_code
    graph.set_entry_point("parse_code")
    
    # 조건부 엣지: parse_code 이후 valid 여부에 따라 분기
    graph.add_conditional_edges(
        "parse_code",           # 이 노드 실행 후
        should_fix,             # 이 함수로 다음 노드 결정
        {
            "fix": "fix_code",      # "fix" 반환 시
            "score": "score_code",  # "score" 반환 시
        }
    )
    
    # fix_code 후에는 다시 parse_code로 (재검증 루프)
    graph.add_edge("fix_code", "parse_code")
    
    # score_code 후에는 종료
    graph.add_edge("score_code", END)
    
    return graph.compile()


# ── 6. 실행 ────────────────────────────────────────────────────

def run_pipeline(html: str) -> HTMLState:
    """파이프라인 실행 진입점"""
    pipeline = build_pipeline()
    
    # 초기 State
    initial_state: HTMLState = {
        "html": html,
        "valid": False,
        "error": "",
        "attempts": 0,
        "components": 0,
        "hooks": 0,
        "jsx_elements": 0,
        "score": 0.0,
    }
    
    result = pipeline.invoke(initial_state)
    return result


# ── 7. 테스트 ──────────────────────────────────────────────────

if __name__ == "__main__":
    
    # 테스트 1: 정상 코드
    print("=" * 50)
    print("테스트 1: 정상 코드")
    print("=" * 50)
    good_html = "const App = () => <div><h1>Hello</h1><p>World</p></div>"
    result = run_pipeline(good_html)
    print(f"결과: valid={result['valid']}, score={result['score']}, attempts={result['attempts']}")
    
    print()
    
    # 테스트 2: 문법 오류 코드
    print("=" * 50)
    print("테스트 2: 문법 오류 코드")
    print("=" * 50)
    bad_html = "const App = () => <div><h1>Hello</h1>"  # 닫는 태그 없음
    result = run_pipeline(bad_html)
    print(f"결과: valid={result['valid']}, score={result['score']}, attempts={result['attempts']}")
