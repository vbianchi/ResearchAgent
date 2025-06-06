# backend/visualize_graph.py
from pathlib import Path
import re

# Important: This script imports from our existing backend modules.
# Ensure your environment is set up correctly for them to be found.
try:
    from backend.langgraph_agent import research_agent_graph
except ImportError as e:
    print("Error: Could not import 'research_agent_graph' from 'backend.langgraph_agent'.")
    print("Please ensure you are running this script from the project's root directory,")
    print(f"and that the backend package is accessible. Original error: {e}")
    exit(1)

def visualize_graph():
    """
    Generates a MermaidJS text definition of the research_agent_graph.
    """
    print("Generating MermaidJS graph definition...")
    try:
        # Get the graph object from the compiled app
        graph = research_agent_graph.get_graph()
        
        # Get the MermaidJS definition as a string
        mermaid_definition = graph.draw_mermaid()
        
        # <<< START MODIFICATION: Fix invalid dashed-arrow syntax >>>
        # Replace "nodeA -. LABEL .-> nodeB" with "nodeA -.->|LABEL| nodeB"
        mermaid_definition = re.sub(
            r"(\w+)\s-\.\s([^ ]+)\s\.->\s(\w+)(;?)",
            r"\1 -.->|\2| \3\4",
            mermaid_definition
        )
        # <<< END MODIFICATION >>>
        
        # Define the output path in the project's root directory
        output_path = Path(__file__).resolve().parent.parent / "research_agent_graph_definition.md"
        
        # Create the content to be written to the file
        file_content = (
            "# Research Agent Graph Definition\n\n"
            "Copy the code block below and paste it into a Mermaid.js viewer like [mermaid.live](https://mermaid.live) "
            "or a supporting Markdown editor to see the visual graph.\n\n"
            "```mermaid\n"
            f"{mermaid_definition}\n"
            "```\n"
        )
        
        # Write the definition to a Markdown file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(file_content)
        
        print(f"\nSuccessfully generated graph definition!")
        print(f"File saved to: {output_path}")
        print("\nTo view the graph, open the file and paste the content inside the '```mermaid' block into [https://mermaid.live](https://mermaid.live)")

    except Exception as e:
        print(f"\nAn error occurred during graph definition generation: {e}")

if __name__ == "__main__":
    visualize_graph()
