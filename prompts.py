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
            ("human", "{question}"),
        ])


    def query_generator_prompt(self):
        return ChatPromptTemplate.from_messages([
            ("system", 
            """
            You are a SQL query generator specialized in water management systems.
            Your role is to analyze user queries and generate the most relevant SQL queries.

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
            

            <Thought process>
            Think step by step. At each step:
            - Thought: Think about how to solve the problem. Plan how to analyze and visualize the data.
            - Action: Specify the tool and input. If you use python_repl_tool, input the entire code.
            - Observation: Observe the result of the tool execution.

            Example:
            Thought:  Extract the relevant data from the `TB_C_RT` dataframe and calculate the average turbidity for each hour on March 11th.
            Action: Use the `python_repl_tool` to execute the code.
            Observation: The code will extract the relevant data, calculate the average turbidity for each hour, and plot the results.
            
         
            Save the python code generated in the 'python_code' field.

            <Final response>
            1. Summarize the data analysis result.
            2. Provide a clear answer to the user's question.
            3. (Optional) Visualization result with the explanation.

            Write the final response in Korean.

            {agent_scratchpad}
            """
        )


    def db_result_prompt(self):
        return ChatPromptTemplate.from_messages([
            ("system",
             """
             
             """),
        ])

    def sql_query_check_prompt():

        pass
