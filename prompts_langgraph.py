from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


class Prompt:
    def table_selection_prompt(self):
        return ChatPromptTemplate.from_messages([
            ("system",
            """
            You are a expert of database.
            Analyze the user's question and determine which table to query.

            Question: {messages[-1].content}

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
            # ("human", "{question}"),
            MessagesPlaceholder(variable_name="messages")
        ])
    

    def sql_query_generation_prompt(self):
        return ChatPromptTemplate.from_messages([
            ("system",
             """
            You are a SQL expert with a strong attention to detail.
            Your task is to generate a SQL query EXCLUSIVELY for the table {table_name}.
            
            Current task: Generate a query for {table_name} table only
            User's question: {messages[-1].content}
            
            Schema Info:
            {schema_info}

            ===== CRITICAL RULES =====
            1. YOU MUST ONLY GENERATE SQL THAT SELECTS FROM THE {table_name} TABLE
            2. YOUR QUERY MUST START WITH "SELECT ... FROM {table_name}"
            3. ANY OTHER TABLE CAN ONLY BE REFERENCED IN SUBQUERIES
            4. NEVER SELECT FROM A DIFFERENT TABLE AS THE MAIN TABLE
            5. TRIPLE CHECK YOUR QUERY BEFORE RETURNING
            6. RETURN ONLY THE RAW SQL QUERY - NO MARKDOWN, NO BACKTICKS, NO EXPLANATIONS

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
            # ("human", "{question}"),
            MessagesPlaceholder(variable_name="messages")
        ])

    def db_result_prompt(self):
        return ChatPromptTemplate.from_messages([
            ("system",
             """
             
             """),
        ])

    def sql_query_check_prompt():

        pass
