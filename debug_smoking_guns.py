import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'src')))
# We need to mock src.evaluation or import it if possible, but let's just copy the relevant logic to debug it
# actually let's try to import it
try:
    from src.evaluation import analyze_smoking_guns
except ImportError:
    # If running from root
    sys.path.append(os.getcwd())
    from src.evaluation import analyze_smoking_guns

def debug_smoking_guns():
    print("Loading CSV...")
    try:
        df = pd.read_csv('output/evaluation_results_full.csv')
        print("Columns:", df.columns.tolist())
        print("First row Faithfulness_Reasoning:", df.iloc[0]['Faithfulness_Reasoning'])
        
        print("\nRunning analyze_smoking_guns...")
        analyze_smoking_guns(df, rag=None)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_smoking_guns()
