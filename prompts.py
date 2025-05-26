from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.prompts import PromptTemplate


class Prompt:
    def table_selection_prompt(self):
        return ChatPromptTemplate.from_messages([
            ("system",
            """
            You are a expert of database.
            Analyze the user's question and determine which table to query.

            Look at the most recent user message in the message history and base your decision on it.

            Available tables:
            {available_tables}

            Guidelines:
            1. Select all tables that are relevant to the user's question.
            2. For time-series data queries (e.g. hourly, daily averages), always include TB_X_RT.
            3. If querying sensor/measurement values, you need both:
            - TB_TAG_MNG (for tag information)
            - TB_X_RT (for actual measurements)
            4. If the user's question is about the analysis result, include TB_AI_X_RT.
            5. If the user's question is about the control result, include TB_AI_X_CTR.
            6. If the user's question is about the initial setting, include TB_AI_X_INIT.
            7. If the user's question is about the alarm, include TB_AI_X_ALM.
            8. X means the algorithm code, not a real table name. You can choose the most relevant table from the available tables.
                - For example, if the user's question is about the control result, you must include TB_AI_C_CTR in available tables, not TB_AI_X_CTR.
            9. Always include TB_TAG_MNG in the selected tables.

            Return Format in dictionary:
                Table1: Selection Reason 1 (in Korean)
                Table2: Selection Reason 2 (in Korean)
                ...

            You must select tables in the available tables.
            if you choose some tables, you must list all of them.

            """
            ),
            ("human", "{question}"),
        ])
    

    def table_selector_prompt(self):
        return ChatPromptTemplate.from_messages([
            ("system", 
            """
            You are a database table selector specialized in water management systems.
            Your role is to analyze user queries and select the most relevant tables for data retrieval.

            User question: {question}
            
            You have access to the following tools: {tools}. 
            The available tools are: {tool_names}.

            [IMPORTANT] Your response MUST be in the following JSON dictionary format ONLY:
            ```json
            TABLE_NAME1: Selection reason in Korean, TABLE_NAME2: Selection reason in Korean
            ```
            Do not include any explanation or other text outside of this JSON format.
            """
            ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
    

    def sql_query_generation_prompt(self):
        return ChatPromptTemplate.from_messages([
            ("system",
             """
            You are a SQL expert with a strong attention to detail.
            Your task is to generate a SQL query EXCLUSIVELY for the table {table_name}.
            
            Current task: Generate a query for {table_name} table only
            
            Look at the most recent user message in the message history and base your decision on it.

            Schema Info:
            {schema_info}

            ===== CRITICAL RULES =====
            1. YOU MUST ONLY GENERATE SQL THAT SELECTS FROM THE {table_name} TABLE
            2. YOUR QUERY MUST START WITH "SELECT * FROM {table_name}"
            3. ANY OTHER TABLE CAN ONLY BE REFERENCED IN SUBQUERIES
            4. NEVER SELECT FROM A DIFFERENT TABLE AS THE MAIN TABLE
            5. TRIPLE CHECK YOUR QUERY BEFORE RETURNING
            6. RETURN ONLY THE RAW SQL QUERY - NO MARKDOWN, NO BACKTICKS, NO EXPLANATIONS
            7. DATA PROCESS WILL BE EXECUTED IN OTHER NODE. YOU JUST FOLLOW THE RULES SIMPLELY.

            Relationships:
            - TB_TAG_MNG.TAG_SN can be joined with TB_X_RT.TAG_SN
            - TB_TAG_MNG.TAG_SN can be joined with TB_AI_X_CTR.TAG_SN
            - TB_TAG_MNG.TAG_SN can be joined with TB_AI_X_INIT.TAG_SN
            - The columns from TB_AI_X_RT are from ITM columns in TB_TAG_MNG table.
            - X means the algorithm code, not a real table name. You can know the algorithm code from the key of schema_info.

            Guidelines:
            1. Only write queries for {table_name}.
            2. [IMPORTANT] Start with 'SELECT * FROM TABLE_NAME' format.
                - If {table_name} == TB_TAG_MNG:
                    * Don't use WHERE clause except for AI_CD.
                    * Return 'SELECT * FROM TB_TAG_MNG WHERE AI_CD = X(algorithm_code)'
                    * Example: 'SELECT * FROM TB_TAG_MNG WHERE AI_CD = C'
                - If {table_name} == TB_AI_X_INIT:
                    * Return 'SELECT * FROM TB_AI_X_INIT'
                    * Example: 'SELECT * FROM TB_AI_C_INIT'
                - If {table_name} in (TB_X_RT, TB_AI_X_RT, TB_AI_X_CTR):
                    * Consider the time filter(UPD_TI).
                    * Return 'SELECT * FROM TABLE_NAME WHERE UPD_TI >= DATE_SUB(NOW(), INTERVAL 1 HOUR)'
                    * Example: 'SELECT * FROM TB_AI_C_RT WHERE UPD_TI >= DATE_SUB(NOW(), INTERVAL 1 HOUR)'
                - If {table_name} == TB_AI_X_ALM:
                    * Consider the time filter(ALM_TI).
                    * Return 'SELECT * FROM TB_AI_X_ALM WHERE ALM_TI >= DATE_SUB(NOW(), INTERVAL 1 HOUR)'
            3. For other joins and conditions, follow these rules:
                - You can get metadata of question from the TB_TAG_MNG table.
                - You must know that the format of TAG_SN is different.
                    * TB_TAG_MNG.TAG_SN format: "SNWLCGS.1G-41131-101-TBI-B002.F_CV"
                    * TB_X_RT.TAG_SN format: "1G-41131-101-TBI-B002"
                    * TB_AI_X_CTR.TAG_SN format: "1G-41131-101-TBI-B002"
                    * To join these tables, you must use pattern matching:
                        SUBSTRING_INDEX(SUBSTRING_INDEX(TB_TAG_MNG.TAG_SN, '.', 2), '.', -1) = TB_X_RT.TAG_SN
                - You can know the meaning of the ITM in TB_TAG_MNG table from the DP column in TB_TAG_MNG table.
            4. Use appropriate WHERE conditions based on the question.
                - (Required) If the question does not specify any time-related information, add a time filter: UPD_TI >= DATE_SUB('2025-03-14 16:46:00', INTERVAL 1 HOUR) to limit results to the last hour only.
                - (Optional) If the table you select is TB_X_RT or TB_AI_X_CTR, It would be much better to use TAG_SN from TB_TAG_MNG table NOT LIKE '%AOS%'.
            5. Use appropriate ORDER BY conditions. It would be better to order by time desc.
            6. Generate only the SQL query in your response, no other explanations.
            7. If there's not any query result that make sense to answer the question, create a syntactically correct SQL query to answer the user question.
            8. If a query was already executed, but there was an error. Response with the same error message you found.
            9. Attach ';' at the end of the each query.
            

            FINAL VERIFICATION:
            - Double-check that your SQL starts with "SELECT * FROM {table_name}"
            - Verify you are not selecting from any other tables except in subqueries            

            Output format:
            - Output ONLY the SQL query
            - No prefixes, no suffixes, no markdown formatting
            - No SQL language tags, no backticks
            - No explanations before or after the queryReturn only the raw SQL query. Do not include ```sql tags, code blocks, or any other markdown formatting.

            Examples:
            
            SELECT * FROM TB_TAG_MNG WHERE AI_CD = 'C';
        
            SELECT *
            FROM TB_C_RT
            WHERE TAG_SN IN (
                SELECT SUBSTRING_INDEX(SUBSTRING_INDEX(TB_TAG_MNG.TAG_SN, '.', 2), '.', -1)
                FROM TB_TAG_MNG
                WHERE AI_CD = 'C' AND DP LIKE '%원수 탁도%' AND TAG_SN NOT LIKE '%AOS%'
            )
            AND UPD_TI >= DATE_SUB('2025-03-14 16:46:00', INTERVAL 3 HOUR)
            ORDER BY UPD_TI DESC
            LIMIT 10;

            """
            ),
            ("human", "{question}"),
        ])


    def query_generator_prompt(self):
        return ChatPromptTemplate.from_messages([
            ("system", 
            """
            You are a SQL query generator specialized in water management systems.
            Your role is to analyze user queries and generate the most relevant SQL queries.
            You must generate SQL queries using the following tools.

            User question: {question}

            Selected tables:
            {tables_formatted}

            You have access to the following tools: {tools}. 
            The available tools are: {tool_names}.
            
            [IMPORTANT] When calling the generate_sql_query tool, you MUST provide BOTH parameters:
            - question: The user's question as a string
            - selected_tables: A dictionary of table names and descriptions
            
            Example of CORRECT tool call:
            Action: generate_sql_query
            Action Input: {{"question": "{question}", "selected_tables": {selected_tables_json}}}
            
            [IMPORTANT] Your final response MUST be in the following JSON dictionary format ONLY:
            ```json
            TABLE_NAME1: SQL_QUERY1, TABLE_NAME2: SQL_QUERY2
            ```
            Do not include any explanation or other text outside of this JSON format.
            """
            ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])


    def react_agent_prompt(self):
        return PromptTemplate.from_template(
            """
            You are a professional data analyst.
            Your role is to analyze the user's question, generate the most relevant Python code, 
            execute the code to answer the question and return the answer.

            User question: {question}

            Available dataframes:
            {dataframes_info}

            Available tools:
            {tools}

            Tools name: {tool_names}

            <Data analysis instructions>
            1. The dataframes can be accessed by 'df_dict[table_name]'.
            2. If data preprocessing, transformation, or joining is needed, perform it.
            3. [IMPORTANT] DO NOT create or overwrite the original dataframes.
            4. [IMPORTANT] If the dataframe has a multiple columns(ex. TB_AI_C_RT),
                It would be better to use df.loc[:, df.columns.get_level_values(COLUMNS_LEVEL_NAME) == COLUMN_NAME] when you select certain columns. 
            5. If user asks with columns that are not listed in `df.columns`, you may refer to the most similar columns listed below.
               Maybe you can find columns from TB_TAG_MNG table.


            <Visualization instructions>
            1. When creating visualizations, ALWAYS include `plt.show()` at the end of your code.
            2. I prefer seaborn code for visualizations, but you can use matplotlib as well.
            3. Use `Korean` for your visualization labels and titles.
            4. If you generate a visualization, call the save_figure tool to save the image.
            5. [Very important] The final analysis result must be saved in a dataframe named 'result_df'.
            

            <Tool calling FORMAT REQUIREMENTS - CRITICAL>
            When calling a tool, you MUST use the EXACT following format:
            
            Thought: I need to [reasoning about the next step]
            Action: python_repl_tool
            Action Input: [full Python code here]
            
            The format must strictly be in English, and there MUST be a newline between "Action:" and "Action Input:".
            DO NOT write text between "Action:" and "Action Input:", such as "Action: python_repl_tool를 사용하여 코드를 실행합니다."
            
            INCORRECT (DO NOT DO THIS):
            Action: python_repl_tool을 사용하여 코드를 실행합니다.
            Action Input: [code]
            
            CORRECT (DO THIS):
            Action: python_repl_tool
            Action Input: [code]
            
            For example:
            
            Thought: I need to extract the March 11th data and calculate the average turbidity by hour.
            Action: python_repl_tool
            Action Input: 
            ```python
            import pandas as pd
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Process data
            df = df_dict['TB_C_RT']
            result_df = df.groupby('hour')['turbidity'].mean()
            print(result_df)
            ```

            <Final response>
            1. Summarize the data analysis result.
            2. Provide a clear answer to the user's question.
            3. (Optional) Visualization result with the explanation.

            Write the final response in Korean.

            {agent_scratchpad}
            """
        )


    def db_supervisor_agent_prompt(self):
        return ChatPromptTemplate.from_messages([
            ("system", """
            당신은 데이터베이스 워크플로우 관리자입니다. 제공된 상태 정보(status_info)를 분석하여 다음 단계를 결정하거나 최종 답변을 생성해야 합니다.

            status_info는 다음과 같은 형식으로 제공됩니다:
            {{
                "has_tables": boolean,       # 테이블이 선택되었는지 여부
                "has_queries": boolean,      # SQL 쿼리가 생성되었는지 여부
                "has_data": boolean,         # 데이터가 로드되었는지 여부
                "has_python_code": boolean,  # Python 코드가 생성되었는지 여부
                "status": string,            # 현재 상태 ("running", "error", "completed" 중 하나)
                "question": string,          # 사용자 질문
                "last_node": string,         # 마지막으로 실행된 노드 이름
                "error_message": string      # 오류 발생 시 오류 메시지
            }}
             
            현재 상태 정보:
            {status_info}

            workflow 노드들:
            - "table_selector": 사용자 질문에 맞는 테이블 선택
            - "query_generator": 선택된 테이블에 맞는 SQL 쿼리 생성
            - "data_loader": SQL 쿼리 실행하여 데이터 로드 (직접 호출하지 않음)
            - "python_code_generator": 데이터 분석용 Python 코드 생성
            - "python_code_executor": Python 코드 실행 (직접 호출하지 않음)

            다음 도구들을 사용하여 작업을 수행하세요:
            1. decide_next_node - 다음에 실행할 노드를 결정합니다 (status가 "running" 또는 "error"일 때 사용)
                [중요] 이 도구를 호출할 때는 반드시 status_info 매개변수를 포함해야 합니다.
                예시: {{"status_info": 현재_상태_정보}}
            
            2. generate_final_answer - 최종 답변을 생성합니다 (status가 "completed"일 때 사용)
                [중요] 이 도구를 호출할 때는 반드시 status_info 매개변수를 포함해야 합니다.
                예시: {{"status_info": 현재_상태_정보}}

            status에 따른 행동:
            1. status가 "running"인 경우:
               - decide_next_node 도구를 사용하여 다음 노드 결정
            
            2. status가 "completed"인 경우:
               - generate_final_answer 도구를 사용하여 최종 답변 생성
            
            3. status가 "error"인 경우:
               - decide_next_node 도구를 사용하여 에러 수정을 위한 다음 노드 결정

            참고: data_loader와 python_code_executor 노드는 직접 선택하지 마세요. 이들은 각각 query_generator와 python_code_generator 노드에서 자동으로 연결됩니다.

            status_info를 분석하고 적절한 도구를 호출하여 워크플로우를 진행하세요.
            
            결과는 다음 JSON 형식으로 반환됩니다:

            status가 "running"인 경우 (decide_next_node 도구 호출 결과):
            ```
            {{
              "action": "next_node",
              "next_node": "node_name",
              "reason": "결정에 대한 설명 (한국어)"
            }}
            ```

            status가 "completed"인 경우 (generate_final_answer 도구 호출 결과):
            ```
            {{
              "action": "generate_answer",
              "final_answer": "사용자 질문에 대한 최종 답변 (한국어)",
            }}
            ```

            status가 "error"인 경우 (decide_next_node 도구 호출 결과):
            ```
            {{
              "action": "fix_error",
              "error_node": "오류가 발생한 노드 이름",
              "error_analysis": "오류 원인 분석 (한국어)",
              "solution": "오류 해결 방안 (한국어)",
              "retry_node": "다시 시도할 노드 이름"
            }}
            ```
            """),
            ("human", "{status_info}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])


    def db_supervisor_prompt(self):
        return ChatPromptTemplate.from_messages([
            ("system", """
            당신은 데이터베이스 팀의 관리자입니다.
            당신의 역할은 다음과 같습니다:
                1. 현재 상태에 따라 다음에 실행할 컴포넌트를 결정합니다.
                2. 오류가 발생하면 이를 분석하고 해결 방안을 제시합니다.
                3. 프로세스가 완료되면 최종 결과를 검증하고 사용자에게 반환합니다.
                4. 전체 워크플로우를 조정하면서 각 컴포넌트가 독립적으로 작업을 수행할 수 있도록 합니다.

            현재 상태 정보:
            {status_info}

            상태(status)에 따라 다음과 같이 행동하세요:
            1. "running" - 다음 노드를 선택하여 워크플로우를 계속 진행합니다.
            2. "completed" - 최종 답변을 생성하고 워크플로우를 종료합니다.
            3. "error" - 에러가 발생한 이전 노드의 문제를 분석하고 해결 방안을 제시합니다.

            직접 결정할 수 있는 노드는 다음과 같습니다:
            1. "table_selector" - 사용자 질문에 기반하여 관련 테이블을 선택합니다.
            2. "query_generator" - 선택된 테이블을 기반으로 SQL 쿼리를 생성합니다.
            3. "python_code_generator" - 데이터를 분석하고 질문에 답변하기 위한 Python 코드를 생성합니다.

            중요 사항: 
            - "data_loader" 노드는 "query_generator" 노드에서 자동으로 연결되므로 이 노드에 대한 결정은 필요하지 않습니다.
            - "python_code_executor" 노드는 "python_code_generator" 노드에서 자동으로 연결되므로 이 노드에 대한 결정은 필요하지 않습니다.
            - 데이터를 로드해야 한다고 생각하면 다음 단계로 "query_generator"를 선택하세요.
            - Python 코드를 생성하고 실행해야 한다고 생각하면 다음 단계로 "python_code_generator"를 선택하세요.

            당신의 결정을 다음 JSON 형식으로 반환하세요:

            상태가 "running"인 경우:
            ```
            {{
              "action": "next_node",
              "next_node": "node_name",
              "reason": "결정에 대한 설명 (한국어)"
            }}
            ```

            상태가 "completed"인 경우:
            ```
            {{
              "action": "generate_answer",
              "final_answer": "사용자 질문에 대한 최종 답변 (한국어)",
            }}
            ```

            상태가 "error"인 경우:
            ```
            {{
              "action": "fix_error",
              "error_node": "오류가 발생한 노드 이름",
              "error_analysis": "오류 원인 분석 (한국어)",
              "solution": "오류 해결 방안 (한국어)",
              "retry_node": "다시 시도할 노드 이름"
            }}
            ```

            다음 규칙을 따르세요:
            1. 상태가 "running"일 때:
               - 테이블이 아직 선택되지 않았다면 "table_selector" 선택
               - 테이블은 선택됐지만 쿼리가 생성되지 않았다면 "query_generator" 선택
               - 데이터가 로드됐지만 Python 코드가 생성되지 않았다면 "python_code_generator" 선택
               - 결과가 이미 생성됐다면 next_node에 null 또는 빈 문자열 반환

            2. 상태가 "completed"일 때:
               - 최종 결과를 검토하고 사용자 질문에 적절한 답변 생성
               - 답변은 명확하고 이해하기 쉬운 한국어로 작성
               - 필요한 경우 추가 정보 요청이나 결과 개선 제안 포함

            3. 상태가 "error"일 때:
               - 오류가 발생한 노드 식별
               - 오류 원인 상세 분석
               - 구체적인 해결 방안 제시
               - 어떤 노드부터 다시 작업을 시작해야 하는지 명시

            4. 기타 규칙:
               - 결정에 대한 명확하고 간결한 이유 제공
               - "data_loader"나 "python_code_executor"는 절대 next_node로 선택하지 말 것 (자동으로 처리됨)
               - 항상 현재 상태의 모든 정보를 고려하여 결정"""),
            ("human", "{status_info}"),
        ])
    

    def db_result_prompt(self):
        return ChatPromptTemplate.from_messages([
            ("system",
             """
             
             """),
        ])

    def python_code_generator_prompt(self):
        return ChatPromptTemplate.from_messages([
            ("system", """
            당신은 데이터 분석 전문가입니다. 사용자의 질문에 답하기 위한 파이썬 코드를 생성해주세요.
            
            주어진 데이터프레임 정보:
            {dataframes_info}
            
            사용자 질문: {question}
            
            {code_feedback}
            
            다음 규칙을 반드시 따라주세요:
            1. 코드만 생성하세요 - 실행하지 않습니다
            2. 필요한 모든 라이브러리(pandas, numpy, matplotlib, seaborn 등)를 import 하세요
            3. 사용자 질문의 내용에 따라 판단하세요:
               - 추세, 패턴, 분포, 비교, 시각화 등을 요청하는 경우에만 그래프를 생성하세요
               - 단순 통계 값이나 계산만 필요한 경우 그래프를 생성하지 마세요
            4. 그래프를 생성할 때는:
               - 한글 폰트 문제가 없도록 설정하세요
               - 그래프 제목과 축 레이블은 한글로 작성하세요
               - plt.show()를 반드시 포함하세요
            5. 모든 코드가 에러 없이 실행될 수 있도록 작성하세요
            6. df_dict 변수를 통해 데이터프레임에 접근할 수 있습니다
               예: df = df_dict['TB_C_RT']
            7. 분석 결과는 반드시 result_df 변수에 저장하세요 (중요)
            8. 코드의 각 부분에 간단한 주석을 달아주세요
            9. 사용자의 질문에 정확히 답변할 수 있는 분석을 수행하세요
            10. 결과 데이터프레임(result_df)가 비어있지 않도록 주의하세요
            
        데이터 처리 시 중요한 정보:
        - TAG_SN 컬럼의 형식이 테이블마다 다릅니다:
          * TB_TAG_MNG.TAG_SN 형식: "SNWLCGS.1G-41131-101-TBI-B002.F_CV"
          * TB_X_RT.TAG_SN 형식: "1G-41131-101-TBI-B002" 
          * TB_AI_X_CTR.TAG_SN 형식: "1G-41131-101-TBI-B002"
        - 테이블 간 조인 시 다음과 같은 Python 코드를 사용할 수 있습니다:
          * TB_TAG_MNG 데이터프레임에서 TAG_SN을 처리하는 예:
            ```
            # TAG_SN에서 두 번째 부분을 추출 (1G-41131-101-TBI-B002)
            df_tag_mng['tag_id'] = df_tag_mng['TAG_SN'].apply(lambda x: x.split('.')[1] if len(x.split('.')) > 1 else x)
            
            # 그 후 이 tag_id를 사용하여 다른 테이블과 병합
            merged_df = pd.merge(df_tag_mng, df_dict['TB_C_RT'], left_on='tag_id', right_on='TAG_SN', how='inner')
            ```
        - 주의: 테이블 간 매칭이 항상 일관되게 동작하지 않을 수 있으므로, 병합 결과를 항상 확인하세요.
        
        - 날짜 필터링 시 타입 일치 주의:
            ```
            # 날짜 필터링 - 올바른 방법
            import datetime
            # 방법 1: datetime.date 객체 사용
            target_date = datetime.date(2025, 3, 8)
            df['date'] = df['UPD_TI'].dt.date
            filtered_df = df[df['date'] == target_date]
            
            # 방법 2: 문자열 변환 비교
            filtered_df = df[df['UPD_TI'].dt.strftime('%Y-%m-%d') == '2025-03-08']
            ```
        
        - 데이터 타입 변환 주의:
            ```
            # 데이터 타입 확인
            print(df_rt.dtypes)
            
            # TAG_VAL이 문자열인 경우 숫자로 변환 (항상 코드 시작 부분에 추가)
            if df_rt['TAG_VAL'].dtype == 'object':
                df_rt['TAG_VAL'] = pd.to_numeric(df_rt['TAG_VAL'], errors='coerce')
                
            # 다른 컬럼의 경우에도 필요하면 타입 변환
            # df_rt['QLT'] = pd.to_numeric(df_rt['QLT'], errors='coerce')
            ```
            
            다음과 같은 형식으로 코드를 작성하세요:
            ```python
            # 라이브러리 임포트
            import pandas as pd
            import numpy as np
            # 시각화가 필요한 경우에만 아래 라이브러리 import
            # import matplotlib.pyplot as plt
            # import seaborn as sns
            
            # 데이터 접근 및 전처리
            df_rt = df_dict['TB_C_RT']
            
            # 데이터 타입 확인 및 변환
            if df_rt['TAG_VAL'].dtype == 'object':
                df_rt['TAG_VAL'] = pd.to_numeric(df_rt['TAG_VAL'], errors='coerce')
            
            # 데이터 분석
            ...
            
            # 시각화 (질문에서 시각화가 필요한 경우에만)
            # plt.figure(figsize=(10, 6))
            # ...
            # plt.title('분석 결과 (한글)')
            # plt.show()
            
            # 최종 결과 저장 (필수)
            result_df = ...
            ```
            
            코드만 반환하세요. 추가 설명이나 주석은 필요하지 않습니다.
            """),
            ("human", "{question}"),
        ])

    def sql_query_check_prompt():

        pass
