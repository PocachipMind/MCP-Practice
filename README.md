# MCP-Practice
MCP 서버를 구현하고 이를 LangGraph와 Claude를 이용하여 활용하는 연습을 합니다.

## 1. FastMCP 활용 MCP 서버 ( 파일 mcp_server.py )

MCP 서버를 FastMCP 클래스로 구성. 가상의 데이터베이스로 부터 정보를 얻는 MCP서버를 간단히 구현해봄.

1. OpenAI 를 쓰기 위한 환경설정 ```load_dotenv()```
2. FastMCP(name="Virtual_DB") 로 서버 인스턴스 생성
3. @mcp.tool을 활용해 가상의 데이터베이스 정보를 찾아보는 Tools 구현
4. mcp.run(transport="stdio")로 서버 실행하면 STDIO를 통해 외부연결 기다림. STDIO = 서버를 별도 프로세스로 실행하고 표준 입출력 파이프로 통신하는 경량 IPC 방식 [(Git)](https://github.com/commitbyrajat/langgraph-mcp-integration) 

```
from mcp.server.fastmcp import FastMCP

import os
import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()

# MCP 서버 생성
mcp = FastMCP(name="Virtual_DB")

@mcp.tool(
    name="file_list",
    description= """데이터 베이스 안에 있는 파일목록을 알려줍니다.
    
    데이터 베이스 안의 내용을 원하는 사용자를 위해 파일을 확인하고 답해주세요.
    
    파일의 이름을 통해 어떤 데이터인지 간략히 알 수 있습니다.
    
    모든 파일은 csv 파일입니다.
    
    Returns:
        list : 파일 목록
    """
)
def file_list() -> list:
    folder_path = "C:/Users/User/Desktop/DeepLearning/MCP_PJ/Virtual_databases"
    files = [
        f
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ]
    return files

@mcp.tool(
    name="csv_analyst",
    description= """데이터 베이스내의 csv 파일을 분석할 때 사용하는 도구입니다.
    
    해당 도구를 사용하기 전 file_list 도구를 통해 분석할 csv 파일의 이름을 확인하세요.
    
    데이터 베이스 내부의 분석하길 원하는 csv 파일 이름과 해당 데이터에 대해 궁금한 정보를 인자로 받습니다.

    Args:
        csvfile (str): 데이터 베이스의 csv 파일 이름
        question (str): 물어볼 질문. ( 예 : "데이터셋의 행과 열의 수를 알려주세요." )
        
    Returns:
        str: question의 대한 응답.
    """
)
def csv_analyst( csvfile: str, question: str ) -> str:
    folder_path = "C:/Users/User/Desktop/DeepLearning/MCP_PJ/Virtual_databases"
    full_path = os.path.join(folder_path, csvfile)
    
    # 확장 가능하도록 모든 csv 읽도록
    try:
        df = pd.read_csv(full_path)  # 기본: utf-8
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(full_path, encoding='cp949')  # 대안 1
        except UnicodeDecodeError:
            df = pd.read_csv(full_path, encoding='utf-8-sig')  # 대안 2

    # OpenAI의 ChatGPT 모델을 초기화
    llm = ChatOpenAI(model="gpt-4o-mini")

    # Pandas DataFrame을 처리할 수 있는 LangChain 에이전트를 생성
    pandas_agent = create_pandas_dataframe_agent(
        llm,  # 초기화된 언어 모델
        df,   # 분석할 DataFrame
        agent_type="openai-tools",  # OpenAI의 도구를 사용하는 에이전트 유형 지정
        verbose=False,              # 상세한 출력 비활성화
        return_intermediate_steps=True,  # 에이전트의 중간 처리 단계 결과 반환 설정
        allow_dangerous_code=True,       # 잠재적으로 위험할 수 있는 코드 실행 허용
    )

    result = pandas_agent.invoke(question)
    
    return result['output']
    

if __name__ == "__main__":
    # MCP 서버 실행 (표준입출력 transport 사용)
    mcp.run(transport="stdio")
```
