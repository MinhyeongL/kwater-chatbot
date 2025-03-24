from langchain_core.prompts import ChatPromptTemplate


class Prompt:
    def table_selection_prompt(self):
        return ChatPromptTemplate.from_messages([
            ("system",
             """
             당신은 데이터베이스 전문가입니다.
             사용자의 질문을 분석하여 어떤 테이블을 조회해야 할지 결정해주세요.

            사용 가능한 테이블:
            - TB_C_RT: 실시간 측정 데이터 테이블. 센서의 실시간 측정값 저장
            - TB_AI_C_RT: AI 분석 결과 테이블. AI 예측값과 분석 결과 저장
            - TB_AI_C_CTR: AI 제어 결과 테이블. AI 제어 명령과 결과 저장
            - TB_TAG_MNG: 태그 관리 테이블. 센서 등의 메타데이터 저장

            응답 형식:
            테이블: [테이블명1, 테이블명2, ...]
            이유: [선택 이유 설명]

            테이블은 반드시 위 목록에 있는 테이블만 선택해야 합니다.
            여러 테이블이 필요한 경우 모두 나열하세요.
            """),
            ("human", "{question}")
        ])

    def sql_query_gen_prompt(self, table_schemas: dict):
        def process_schema_for_prompt(table_name, schema_text):
            """테이블별로 스키마 정보 처리"""
            if table_name == "TB_AI_C_RT":
                # JSON이 많은 테이블은 샘플 데이터 제외
                return schema_text.split("/*")[0].strip()
            else:
                # 다른 테이블은 샘플 데이터 포함
                return schema_text

        schema_info = ""
        for table, schema in table_schemas.items():
            schema_info += f"\n## {table} Table:\n{process_schema_for_prompt(table, schema)}\n"

        system_message = f"""
             You are a SQL expert with a strong attention to detail.

             You can define SQL queries, analyze queries results and interpretate query results to response an answer.

             You can use the following table schemas and relationships to generate a SQL query:
             Schema Info:
             {schema_info}

             Relationships:
             - TB_TAG_MNG.TAG_SN can be joined with TB_C_RT.TAG_SN
             - TB_TAG_MNG.TAG_SN can be joined with TB_AI_C_CTR.TAG_SN
             - The columns from TB_AI_C_RT are from ITM columns in TB_TAG_MNG table.

             Guidelines:
             1. Write a SELECT query that matches the question.
                - If you select from TB_C_RT, TB_AI_C_RT or TB_AI_C_CTR, select UPD_TI together.
             2. Use JOIN when necessary.
                - You can get metadata of question from the TB_TAG_MNG table.
                For example, You can use the TAG_SN column from TB_TAG_MNG to join with TB_C_RT or TB_AI_C_CTR tables,
                as it serves as a foreign key relationship between these tables.
                - You must know that the format of TAG_SN is different.
                    * TB_TAG_MNG.TAG_SN format: "SNWLCGS.1G-41131-101-TBI-B002.F_CV"
                    * TB_C_RT.TAG_SN format: "1G-41131-101-TBI-B002"
                    * TB_AI_C_CTR.TAG_SN format: "1G-41131-101-TBI-B002"
                    * To join these tables, you must use pattern matching:
                        SUBSTRING_INDEX(SUBSTRING_INDEX(TB_TAG_MNG.TAG_SN, '.', 2), '.', -1) = TB_C_RT.TAG_SN
                - You can know the meaning of the ITM in TB_TAG_MNG table from the DP column in TB_TAG_MNG table.
                - The columns from TB_AI_C_RT are from ITM columns in TB_TAG_MNG table.
             3. Use clear WHERE conditions.
                - (Required) When you search in TB_TAG_MNG table, It would be much better to use AI_CD = 'C' and DP LIKE '%Question%'.
                - (Optional) If the table you select is TB_C_RT or TB_AI_C_CTR, It would be much better to use TAG_SN from TB_TAG_MNG table NOT LIKE '%AOS%'
                - (Required) If the question does not specify any time-related information, add a time filter: UPD_TI >= DATE_SUB('2025-03-14 16:46:00', INTERVAL 1 HOUR) to limit results to the last hour only.
             4. Use clear ORDER BY conditions.
                - If you select from TB_C_RT, TB_AI_C_RT or TB_AI_C_CTR, order by UPD_TI DESC.
             5. Do not include comments in the query.
             6. Provide only the SQL query in your response, no other explanations.
             7. Use the exact column names as specified.
             7. If there's not any query result that make sense to answer the question, create a syntactically correct SQL query to answer the user question. DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
             8. If a query was already executed, but there was an error. Response with the same error message you found.
             9. Add LIMIT 10 if there is no time filter.

             Return ONLY the raw SQL query. Do not include ```sql tags, code blocks, or any other markdown formatting.


             For Example:

             SELECT UPD_TI, TAG_VAL
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

        template = ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", "{question}"),
        ])

        return template

    def db_result_prompt(self):
        return ChatPromptTemplate.from_messages([
            ("system",
             """
             
             """),
        ])

    def sql_query_check_prompt():

        pass
