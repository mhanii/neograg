import os
import json
import asyncio
import neo4j
from dotenv import load_dotenv
from typing import Any, Optional, Union, List
from neo4j_graphrag.retrievers import VectorRetriever, VectorCypherRetriever
from neo4j_graphrag.indexes import create_vector_index
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from neo4j_graphrag.generation import RagTemplate
from neo4j_graphrag.generation.graphrag import GraphRAG
from providers.gemini import GeminiLLM, GeminiEmbeddings

# Load environment variables
load_dotenv('.env', override=True)
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')



def load_prompt_template(path='prompts/graph-builder.txt'):
    """Load prompt template from file"""
    with open(path, 'r') as f:
        return f.read()


def setup_graph_components():
    """Initialize Neo4j driver, custom Gemini LLM, and embeddings"""
    driver = neo4j.GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    
    # Custom Gemini LLM for KG extraction
    ex_llm = GeminiLLM(model_name="gemini-2.5-flash", model_params={"temperature": 0})
    
    # Custom Gemini embeddings (3072 dimensions)
    embedder = GeminiEmbeddings()
    
    return driver, ex_llm, embedder


def get_node_and_relation_types():
    """Define entity and relationship types for extraction"""
    basic_node_labels = ["Object", "Entity", "Group", "Person", "Organization", "Place"]
    academic_node_labels = ["ArticleOrPaper", "PublicationOrJournal"]
    medical_node_labels = ["Anatomy", "BiologicalProcess", "Cell", "CellularComponent", 
                           "CellType", "Condition", "Disease", "Drug",
                           "EffectOrPhenotype", "Exposure", "GeneOrProtein", "Molecule",
                           "MolecularFunction", "Pathway"]
    
    node_labels = basic_node_labels + academic_node_labels + medical_node_labels
    
    rel_types = ["ACTIVATES", "AFFECTS", "ASSESSES", "ASSOCIATED_WITH", "AUTHORED",
        "BIOMARKER_FOR", "CAUSES", "CITES", "CONTRIBUTES_TO", "DESCRIBES", "EXPRESSES",
        "HAS_REACTION", "HAS_SYMPTOM", "INCLUDES", "INTERACTS_WITH", "PRESCRIBED",
        "PRODUCES", "RECEIVED", "RESULTS_IN", "TREATS", "USED_FOR"]
    
    return node_labels, rel_types


async def build_knowledge_graph(driver, ex_llm, embedder, pdf_paths):
    """Extract knowledge graph from PDF documents"""
    prompt_template = load_prompt_template()
    node_labels, rel_types = get_node_and_relation_types()
    
    kg_builder = SimpleKGPipeline(
        llm=ex_llm,
        driver=driver,
        text_splitter=FixedSizeSplitter(chunk_size=500, chunk_overlap=100),
        embedder=embedder,
        entities=node_labels,
        relations=rel_types,
        prompt_template=prompt_template,
        from_pdf=True
    )
    
    results = []
    for path in pdf_paths:
        print(f"Processing: {path}")
        result = await kg_builder.run_async(file_path=path)
        results.append(result)
        print(f"Result: {result}")
    
    return results


def setup_vector_index(driver):
    """Create vector index for embeddings (3072 dims for Gemini)"""
    create_vector_index(
        driver, 
        name="text_embeddings", 
        label="Chunk",
        embedding_property="embedding", 
        dimensions=3072,
        similarity_fn="cosine"
    )


def create_retrievers(driver, embedder):
    """Setup both vector-only and vector+cypher retrievers"""
    
    # Vector-only: semantic similarity search
    vector_retriever = VectorRetriever(
        driver,
        index_name="text_embeddings",
        embedder=embedder,
        return_properties=["text"],
    )
    
    # Vector + Cypher: semantic search + graph traversal
    vc_retriever = VectorCypherRetriever(
        driver,
        index_name="text_embeddings",
        embedder=embedder,
        retrieval_query="""
    WITH node AS chunk
    MATCH (chunk)<-[:FROM_CHUNK]-()-[relList:!FROM_CHUNK]-{1,2}()
    UNWIND relList AS rel
    
    WITH collect(DISTINCT chunk) AS chunks, 
      collect(DISTINCT rel) AS rels
    
    RETURN '=== text ===\n' + apoc.text.join([c in chunks | c.text], '\n---\n') + '\n\n=== kg_rels ===\n' +
      apoc.text.join([r in rels | startNode(r).name + ' - ' + type(r) + '(' + coalesce(r.details, '') + ')' +  ' -> ' + endNode(r).name ], '\n---\n') AS info
    """
    )
    
    return vector_retriever, vc_retriever


def test_retrievers(vector_retriever, vc_retriever):
    """Test both retriever types with sample query"""
    query = "How is precision medicine applied to Lupus?"
    
    print("\n=== VECTOR RETRIEVER TEST ===")
    vector_res = vector_retriever.get_search_results(query_text=query, top_k=3)
    for i in vector_res.records: 
        print("====\n" + json.dumps(i.data(), indent=4))
    
    print("\n=== VECTOR + CYPHER RETRIEVER TEST ===")
    vc_res = vc_retriever.get_search_results(query_text=query, top_k=3)
    
    kg_rel_pos = vc_res.records[0]['info'].find('\n\n=== kg_rels ===\n')
    print("# Text Chunk Context:")
    print(vc_res.records[0]['info'][:kg_rel_pos])
    print("\n# KG Context From Relationships:")
    print(vc_res.records[0]['info'][kg_rel_pos:])


def setup_rag_pipelines(vector_retriever, vc_retriever):
    """Create RAG instances for both retriever types"""
    llm = GeminiLLM(model_name="gemini-1.5-pro", model_params={"temperature": 0.0})
    
    rag_template = RagTemplate(
        template='''Answer the Question using the following Context. Only respond with information mentioned in the Context. Do not inject any speculative information not mentioned. 

# Question:
{query_text}
 
# Context:
{context}

# Answer:
''', 
        expected_inputs=['query_text', 'context']
    )
    
    v_rag = GraphRAG(llm=llm, retriever=vector_retriever, prompt_template=rag_template)
    vc_rag = GraphRAG(llm=llm, retriever=vc_retriever, prompt_template=rag_template)
    
    return v_rag, vc_rag


def compare_rag_approaches(v_rag, vc_rag):
    """Compare vector-only vs vector+cypher RAG responses"""
    print("\n=== RAG COMPARISON TEST ===")
    q = "How is precision medicine applied to Lupus? provide in list format."
    
    print(f"\nVector Response:\n{v_rag.search(q, retriever_config={'top_k':5}).answer}")
    print("\n===========================\n")
    print(f"Vector + Cypher Response:\n{vc_rag.search(q, retriever_config={'top_k':5}).answer}")


def detailed_query_analysis(v_rag, vc_rag):
    """Run detailed query and inspect retrieved contexts"""
    print("\n=== DETAILED QUERY ANALYSIS ===")
    q = "Can you summarize systemic lupus erythematosus (SLE)? including common effects, biomarkers, and treatments? Provide in detailed list format."
    
    v_rag_result = v_rag.search(q, retriever_config={'top_k': 5}, return_context=True)
    vc_rag_result = vc_rag.search(q, retriever_config={'top_k': 5}, return_context=True)
    
    print(f"\nVector Response:\n{v_rag_result.answer}")
    print("\n===========================\n")
    print(f"Vector + Cypher Response:\n{vc_rag_result.answer}")
    
    print("\n=== Vector Context Items ===")
    for i in v_rag_result.retriever_result.items: 
        print(json.dumps(eval(i.content), indent=1))
    
    print("\n=== Biomarker Relationships from Graph ===")
    vc_ls = vc_rag_result.retriever_result.items[0].content.split('\\n---\\n')
    for i in vc_ls:
        if "biomarker" in i.lower(): 
            print(i)
    
    print("\n=== Treatment Relationships from Graph ===")
    for i in vc_ls:
        if "treat" in i.lower(): 
            print(i)


def comprehensive_query_test(v_rag, vc_rag):
    """Test with comprehensive query covering multiple aspects"""
    print("\n=== COMPREHENSIVE QUERY TEST ===")
    q = "Can you summarize systemic lupus erythematosus (SLE)? including common effects, biomarkers, treatments, and current challenges faced by Physicians and patients? provide in list format with details for each item."
    
    print(f"\nVector Response:\n{v_rag.search(q, retriever_config={'top_k': 5}).answer}")
    print("\n===========================\n")
    print(f"Vector + Cypher Response:\n{vc_rag.search(q, retriever_config={'top_k': 5}).answer}")


async def main():
    """Main pipeline orchestration"""
    print("=== INITIALIZING COMPONENTS ===")
    driver, ex_llm, embedder = setup_graph_components()
    
    pdf_paths = [
        'truncated-pdfs/pone.0104830.pdf', 
    ]
    
    print("\n=== BUILDING KNOWLEDGE GRAPH ===")
    await build_knowledge_graph(driver, ex_llm, embedder, pdf_paths)
    
    print("\n=== SETTING UP VECTOR INDEX ===")
    setup_vector_index(driver)
    
    print("\n=== CREATING RETRIEVERS ===")
    vector_retriever, vc_retriever = create_retrievers(driver, embedder)
    
    print("\n=== TESTING RETRIEVERS ===")
    test_retrievers(vector_retriever, vc_retriever)
    
    print("\n=== SETTING UP RAG PIPELINES ===")
    v_rag, vc_rag = setup_rag_pipelines(vector_retriever, vc_retriever)
    
    print("\n=== RUNNING RAG TESTS ===")
    compare_rag_approaches(v_rag, vc_rag)
    detailed_query_analysis(v_rag, vc_rag)
    comprehensive_query_test(v_rag, vc_rag)
    
    driver.close()
    print("\n=== PIPELINE COMPLETE ===")


if __name__ == "__main__":
    asyncio.run(main())