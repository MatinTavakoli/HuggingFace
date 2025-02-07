import nbformat
import argparse
import sys
import io
import traceback

def execute_cell(notebook_path, cell_index):
    """Loads a Jupyter notebook and executes a specific cell by index, capturing output."""
    try:
        # Load the notebook
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        # Validate the cell index
        if cell_index < 0 or cell_index >= len(nb.cells):
            print(f"Error: Cell index {cell_index} is out of range (0-{len(nb.cells)-1}).")
            return

        cell = nb.cells[cell_index]

        # Ensure it's a code cell
        if cell.cell_type != "code":
            print(f"Skipping cell {cell_index}: Not a code cell.")
            return

        # Capture stdout and stderr
        output_buffer = io.StringIO()
        sys.stdout = output_buffer
        sys.stderr = output_buffer  # Capture errors too

        # Create a local execution dictionary
        local_vars = {}

        # Get the source code of the cell
        code = cell.source.strip()

        print(f"\n--- Executing Cell {cell_index} ---\n{code}\n")

        try:
            # Execute the cell code
            exec(code, globals(), local_vars)
        except Exception as exec_error:
            print(f"\n--- Error in Cell {cell_index} ---\n{traceback.format_exc()}")

        # Capture the last expression result (if any)
        last_line = code.split("\n")[-1].strip()
        if last_line and not last_line.startswith("print"):
            try:
                result = eval(last_line, globals(), local_vars)
                print(result)  # Explicitly print the last expression
            except Exception:
                pass  # Ignore if not evaluable

        # Reset stdout/stderr
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

        # Print captured output
        cell_output = output_buffer.getvalue()
        print(f"\n--- Output of Cell {cell_index} ---\n{cell_output}")

    except Exception as e:
        print(f"Error executing cell {cell_index}: {e}")
        print(traceback.format_exc())  # Print full error traceback for debugging

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a specific cell from a Jupyter Notebook.")
    parser.add_argument("-f", "--file", required=True, help="Path to the Jupyter Notebook (.ipynb)")
    parser.add_argument("-c", "--cell", type=int, required=True, help="Cell index to execute (0-based)")

    args = parser.parse_args()
    
    execute_cell(args.file, args.cell)

