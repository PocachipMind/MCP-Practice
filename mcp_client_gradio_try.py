import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI 

import gradio as gr
from typing import List, Tuple
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
            
            example_questions = [
                "데이터 베이스안의 정보에서 사용자 ID 1의 이름이 뭔지 알려줘.",
                "데이터 베이스에 LG 전자 종가 데이터가 있어?",
                "데이터 베이스의 삼성 전자 데이터에서 종가가 가장 높았던 날짜를 알려줘."
            ]
            
            async def chat( message: str, history: List[Tuple[str, str]]) -> str:
                try:
                    query = {"messages": [HumanMessage(content=message)]}
                    
                    result = await agent.ainvoke(query, config=config)
                    
                    if "messages" in result:
                        # 메시지 로깅
                        for msg in result["messages"]:
                            msg.pretty_print() # 터미널에 출력

                        last_message = result["messages"][-1]
                        if isinstance(last_message, AIMessage):
                            response = last_message.content
                    else:
                        response =  "응답을 생성하지 못했습니다."

                except Exception as e:
                    print(f"Error occurred: {str(e)}")
                    response = "죄송합니다. 응답을 생성하는 동안 오류가 발생했습니다. 다시 시도해 주세요."
                
                return response
            
            # ChatInterface 생성
            demo = gr.ChatInterface(
                fn=chat,
                title="데이터 베이스 분석 Agent",
                description="데이터 베이스 내부 자료에 대해 물어보세요!",
                examples=example_questions,
                theme=gr.themes.Soft()
            )

            # Gradio 앱 실행
            demo.launch()
                        
                

# 비동기 컨텍스트 실행
if __name__ == "__main__":
    asyncio.run(main())
