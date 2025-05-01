# MCP-Practice
MCP 서버를 구현하고 이를 LangGraph와 Claude를 이용하여 활용하는 연습을 합니다.

![image](https://github.com/user-attachments/assets/af6c4c03-929e-4b6a-9104-3ca8be651daa)

**서버 :** mcp_server.py

**클라이언트 :** mcp_client_local_terminal.py

**환경 :** MCP-Project.yaml ( 아나콘다 가상환경 활용 )
<br>

## 1. FastMCP 활용 MCP 서버 ( 파일 mcp_server.py )

**툴 목록 :** 
- 데이터 베이스 파일 목록 확인 도구
- 데이터 베이스 내부 파일(csv) 분석 도구
  
<br>

MCP 서버를 FastMCP 클래스로 구성. 

가상의 데이터베이스로 부터 정보를 얻는 MCP서버를 간단히 구현.

<br>

1. OpenAI 를 쓰기 위한 환경설정 ```load_dotenv()```
2. ```FastMCP(name="Virtual_DB")``` 로 서버 인스턴스 생성
3. ```@mcp.tool```을 활용해 가상의 데이터베이스 정보를 찾아보는 Tools 등록 [(git)](https://github.com/modelcontextprotocol/python-sdk#:~:text=mcp%20%3D%20FastMCP%28)
4. ```mcp.run(transport="stdio")```로 서버 실행하여 STDIO를 통해 외부연결 기다림. STDIO = 서버를 별도 프로세스로 실행하고 표준 입출력 파이프로 통신하는 경량 IPC 방식 [(Git)](https://github.com/commitbyrajat/langgraph-mcp-integration) 

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

<br>

### MCP Inspector

터미널 상에서 ```mcp dev "파일경로"``` 를 통해 서버가 잘 동작하는지 Test가 가능합니다. 

( 저의 경우 ```mcp dev "C:\Users\User\Desktop\DeepLearning\mcp_PJ\mcp_server.py"``` )

<img src="https://github.com/user-attachments/assets/eee7e34d-a057-4078-8263-7d1454291b3a" width="85%" height="85%"/>

위에 보이는 MCP Inspector 사이트 에서 Tools의 목록과 Test를 해볼 수 있습니다.

<br>

## 2. Client : Terminal을 통한 대화 ( 파일 mcp_client_local_terminal.py )

MCP 서버의 툴을 호출하여 사용하기.

MCP 서버에 클라이언트 세션으로 연결한 후, 서버가 노출한 툴을 불러와서 에이전트의 도구로 활용.

stdio_client 함수를 사용하여 STDIO 방식 MCP 서버 프로세스를 실행 및 연결.

ClientSession을 통해 통신 세션을 관리.

session.initialize()로 세션을 초기화한 뒤, load_mcp_tools(session) 함수를 이용해 서버의 툴 목록을 가져와 LangChain 호환 툴 객체로 변환.

```
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI 
from langchain_core.messages import HumanMessage, AIMessage

from langgraph.checkpoint.memory import MemorySaver

from dotenv import load_dotenv
load_dotenv()

svrpath = "./mcp_server.py"
    
async def main():
    # 1. MCP 서버 프로세스를 STDIO 모드로 실행하도록 파라미터 설정
    server_params = StdioServerParameters(
        command="python",
        args=[svrpath]  # MCP 서버 스크립트 경로
    )
    # 2. MCP 서버에 STDIO 클라이언트로 접속하여 세션 시작
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # MCP 세션 초기화 (서버 메타데이터 교환)
            await session.initialize()
            # 3. 서버로부터 사용 가능한 툴 불러오기 (툴 메타데이터 조회)
            tools = await load_mcp_tools(session)
            # 4. LLM 모델과 툴을 포함한 LangGraph 에이전트 생성
            model = ChatOpenAI(model="gpt-4")  # OpenAI GPT-4 모델 (API 키 필요)
            memory = MemorySaver()
            config = {"configurable":{"thread_id":"1"}}
            agent = create_react_agent(model, tools=tools, checkpointer=memory)
            
            print("채팅을 시작합니다. 채팅 종료를 원한다면 '테스트 종료'를 입력해주세요.\n")
            text = input("채팅 내용 입력 : ")
            
            # 5. 자연어 질의로 에이전트 실행 (에이전트가 툴 호출하여 답변 생성)
            while text != "테스트 종료":

                query = {"messages": [HumanMessage(content=text)]}
                result = await agent.ainvoke(query, config=config)
                
                print("\n답변 생성 완료\n")
                while text != '0' and text != '1':
                    text = input("'1' -> AI 답변만 보기, '0' -> 모든 메세지 보기\n 입력 : ")
                    
                    # 결과 출력
                    if text == '0' :
                        for message in result["messages"]:
                            message.pretty_print()
                    elif text == '1' :
                        result["messages"][-1].pretty_print()
                    else:
                        print("\n'0' 아니면 '1'을 입력하세요.")
                    print()
                        
                text = input("채팅 내용 입력 : ")
            
# 비동기 컨텍스트 실행
if __name__ == "__main__":
    asyncio.run(main())
```

1. ```StdioServerParameters```에 MCP 서버 스크립트와 실행 방식을 지정
2. ```stdio_client``` 컨텍스트를 통해 MCP 서버를 서브프로세스로 실행하면서 표준입출력 채널을 얻음. 그런 다음 ClientSession을 열고 ```session.initialize()```를 호출하여 서버와 초기 통신을 수행. 이 단계에서 클라이언트는 MCP 서버의 능력(capabilities) 정보를 받아 세션을 설정.
3. ```load_mcp_tools(session)``` 호출 시 세션을 통해 서버의 툴 목록 요청이 전달되고, 서버에 등록된 툴 들의 정보가 표준 규격에 맞는 서술로 반환. 이 정보가 LangChain의 Tool 객체로 변환되어 tools 리스트에 담기는데, 내부적으로 MCP 서버로부터 툴들의 이름, 입력 타입/출력 타입, 설명 등이 전달되어 LangChain 툴로 생성.
4. OpenAI(ChatOpenAI)을 사용해 LangGraph의 ReAct 에이전트를 생성. ```create_react_agent``` 사용, Memory 기능 구현. 보다 구체적인 LangGraph WorkFlow도 지정할 수 있으나 현 연습에서는 MCP 구현에 치중했기에 prebuilt된 Graph 사용.
5. 잘 동작하는지 테스트 하기 위한 Terminal을 통한 대화 로직 구현

<br>

### Test :

"데이터 베이스 안의 정보에서 사용자 ID 1의 이름이 뭔지 알려줘." 라고 질문해보겠습니다.

![image](https://github.com/user-attachments/assets/4c028958-1050-4cc6-827c-619155da70c2)

보면 MCP에 정의된 Tool을 잘 호출하여 LangGraph에 적용된 것을 알 수 있습니다.

ClientSession 을 통해 도구를 호출하고 사용하는 것이므로 session이 닫히면 Tool들을 사용할 수 없고 Graph는 비동기 동작이 필요한 특징이 있습니다.

![VID_20250501_093121-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/3f53a28e-f266-4df2-835a-0f7e2b0eadbc)

<br>

## 3. MCP 서버 Desktop Claude 연결

제작한 MCP 서버를 클로드에 연결해봅니다.

### 1. Desktop Claude를 설치하고 실행합니다.

<img src="https://github.com/user-attachments/assets/fcbe9357-7c50-4d52-bb9e-0cd7d54e99e6" width="70%" height="70%"/>

### 2. 좌측 상단 메뉴를 통해 ```claude_desktop_config.json```을 엽니다.

<img src="https://github.com/user-attachments/assets/fcbe9357-7c50-4d52-bb9e-0cd7d54e99e6" width="70%" height="70%"/>

<img src="https://github.com/user-attachments/assets/74a9afc0-f685-4468-af77-f75154842659" width="70%" height="70%"/>

<img src="https://github.com/user-attachments/assets/321a518d-d179-40eb-89a3-86ed2e42ad9b" width="70%" height="70%"/>

<img src="https://github.com/user-attachments/assets/6f43c8e2-475d-45b3-9adc-67796f75aa08" width="70%" height="70%"/>

### 3. 다음과 같이 서버를 추가합니다.

<img src="https://github.com/user-attachments/assets/e11695d9-0ede-4ea4-bc6d-91b267b54237" width="70%" height="70%"/>

저의 경우 다음과 같이 입력했습니다. 가상환경을 활용하기 위해 Python 경로는 다음과 같습니다.
```
{
    "mcpServers": {
        "Virtual_DB": {
            "command": "C:\\Users\\User\\anaconda3\\envs\\MCP-Project\\python.exe",
            "args": [
                "C:\\Users\\User\\Desktop\\DeepLearning\\mcp_PJ\\mcp_server.py"
            ]
        }
    }
}
```

만약 두 개 이상의 서버를 연결하고자 한다면 다음과 같은 형식으로 작성하세요
```
{
    "mcpServers": {
        "Virtual_DB": {
            "command": "C:\\Users\\User\\anaconda3\\envs\\MCP-Project\\python.exe",
            "args": [
                "C:\\Users\\User\\Desktop\\DeepLearning\\mcp_PJ\\mcp_server.py"
            ]
        },
        "another_Server": {
            "command": "C:\\Users\\User\\anaconda3\\envs\\MCP-Project\\python.exe",
            "args": [
                "C:\\Users\\User\\Desktop\\DeepLearning\\mcp_PJ\\another_Server.py"
            ]
        },
    }
}
```


### 4. 클로드를 완전히 종료했다가 다시 킵니다.

클로드를 다시 키면 망치버튼이 활성화 되어있습니다.

![image](https://github.com/user-attachments/assets/2494834e-d2b7-41ef-bebc-cb1a087e9bd8)

![image](https://github.com/user-attachments/assets/09a5034a-b75a-4a0c-adf2-295b445896cb)

이제 대화를 하면 내가 만든 도구를 사용할 수 있는 클로드와 대화를 합니다.
