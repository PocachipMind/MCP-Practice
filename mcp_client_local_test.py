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
            # 5. 자연어 질의로 에이전트 실행 (에이전트가 툴 호출하여 답변 생성)
            query = {"messages": "데이터 베이스안의 정보에서 사용자 ID 1의 이름이 뭔지 알려줘"}
            query = {"messages": [HumanMessage(content="데이터 베이스안의 정보에서 사용자 ID 1의 이름이 뭔지 알려줘")]}
            result = await agent.ainvoke(query, config=config)
            # 결과 출력
            for message in result["messages"]:
                message.pretty_print()
            
            query = {"messages": "그러면 그 김민수의 나이는?"}
            result = await agent.ainvoke(query, config=config)
            # 결과 출력
            for message in result["messages"]:
                message.pretty_print()
                

# 비동기 컨텍스트 실행
if __name__ == "__main__":
    asyncio.run(main())
