from langchain_teddynote.graphs import visualize_graph
from langgraph.graph import START, END
from langgraph.graph import StateGraph
from abc import abstractmethod, ABCMeta
import operator

from nodes import table_selector_node, query_generator_node, data_loader_node, python_code_generator_node, python_code_executor_node, db_supervisor_node


class LangGraphBuilder(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, State):
        pass
    
    @abstractmethod
    def build_graph(self, node_name, node_function):
        pass


class DB_LangGraphBuilder(LangGraphBuilder):
    def __init__(self, State):
        super().__init__(State)
        self.State = State
        self.builder = StateGraph(State)

    def build_graph(self):
        # 노드 추가
        self.builder.add_node("DB Supervisor", db_supervisor_node)
        self.builder.add_node("table_selector", table_selector_node)
        self.builder.add_node("query_generator", query_generator_node)
        self.builder.add_node("data_loader", data_loader_node)
        self.builder.add_node("python_code_generator", python_code_generator_node)
        self.builder.add_node("python_code_executor", python_code_executor_node)

        # 시작점 설정
        self.builder.set_entry_point("DB Supervisor")


        # DB Supervisor와 다른 노드 간의 엣지 설정
        self.builder.add_edge("table_selector", "DB Supervisor")

        # query_generator와 data_loader 직접 연결
        self.builder.add_edge("query_generator", "data_loader")
        self.builder.add_edge("data_loader", "DB Supervisor")

        # python_code_generator와 python_code_executor 직접 연결
        self.builder.add_edge("python_code_generator", "python_code_executor")
        self.builder.add_edge("python_code_executor", "DB Supervisor")

        # 조건부 라우팅 로직 (DB Supervisor가 결정)
        def route_from_supervisor(state):
            # 완료 상태이거나 에러 상태일 경우 종료
            if getattr(state, "completed", False) or state.status == "error":
                return END
            
            # 다음 노드가 명시적으로 지정된 경우
            next_node = getattr(state, "next", None)
            if next_node == "table_selector":
                return "table_selector"
            elif next_node == "query_generator":
                return "query_generator"
            elif next_node == "python_code_generator":
                return "python_code_generator"
            else:
                # 기본적으로 종료
                return END

        # DB Supervisor에서 조건부 라우팅 추가
        self.builder.add_conditional_edges(
            "DB Supervisor",
            route_from_supervisor,
            {
                "table_selector": "table_selector",
                "query_generator": "query_generator",
                "python_code_generator": "python_code_generator",
                END: END
            }
        )

        self.graph = self.builder.compile()

        return self.graph

    def visualize_graph(self):
        visualize_graph(self.graph)


# 상태 라우터 함수들
def has_error(state) -> str:
    """오류가 있는지 확인"""
    if state.error:
        return "error"
    return "continue"

def has_selected_tables(state) -> str:
    """테이블이 선택되었는지 확인"""
    if state.selected_tables:
        return "has_tables"
    return "needs_tables"

def has_generated_queries(state) -> str:
    """쿼리가 생성되었는지 확인"""
    if state.generated_queries:
        return "has_queries"
    return "needs_queries"

def has_loaded_data(state) -> str:
    """데이터가 로드되었는지 확인"""
    if state.df_dict:
        return "has_data"
    return "needs_data"