import json
import os
import sys
import traceback
import io
import contextlib
import base64

def execute_notebook(nb_path):
    print(f"Executing notebook: {nb_path}...")
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cwd = os.path.dirname(os.path.abspath(nb_path))
    old_cwd = os.getcwd()
    os.chdir(cwd)

    global_env = {
        'display': lambda x: print(x)
    }
    code_count = 1
    total_code_cells = sum(1 for c in nb['cells'] if c['cell_type'] == 'code')
    print(f"Total code cells to execute: {total_code_cells}")

    for idx, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            raw_code = ''.join(cell.get('source', []))
            exec_lines = []
            for line in raw_code.split('\n'):
                if line.strip().startswith('%'):
                    continue
                exec_lines.append(line)
            clean_code = '\n'.join(exec_lines)

            print(f"Executing code cell {code_count}/{total_code_cells} (Cell Index {idx})...", end=' ')
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()

            try:
                plt.close('all')
                with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
                    exec(clean_code, global_env)

                out_text = stdout_capture.getvalue()
                err_text = stderr_capture.getvalue()
                outputs = []
                if out_text:
                    outputs.append({
                        "output_type": "stream",
                        "name": "stdout",
                        "text": [line + '\n' for line in out_text.splitlines()]
                    })
                if err_text:
                    clean_err = [line for line in err_text.splitlines() if "FutureWarning" not in line and "UserWarning" not in line]
                    if clean_err:
                        outputs.append({
                            "output_type": "stream",
                            "name": "stderr",
                            "text": [line + '\n' for line in clean_err]
                        })

                fig_nums = plt.get_fignums()
                for fig_num in fig_nums:
                    fig = plt.figure(fig_num)
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', bbox_inches='tight')
                    buf.seek(0)
                    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
                    outputs.append({
                        "output_type": "display_data",
                        "data": {
                            "image/png": img_b64,
                            "text/plain": ["<Figure size ...>"]
                        },
                        "metadata": {}
                    })
                    plt.close(fig)

                cell['execution_count'] = code_count
                cell['outputs'] = outputs
                print("SUCCESS")
            except Exception as e:
                print(f"FAILED with error: {e}")
                traceback.print_exc()
                cell['execution_count'] = code_count
                cell['outputs'] = [{
                    "output_type": "error",
                    "ename": type(e).__name__,
                    "evalue": str(e),
                    "traceback": traceback.format_exc().splitlines()
                }]
                os.chdir(old_cwd)
                raise e

            code_count += 1

    os.chdir(old_cwd)
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

    print(f"\nAll {total_code_cells} code cells executed successfully without errors!\n")

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    execute_notebook(os.path.join(base_dir, "Sample CV Submission Template.ipynb"))
    execute_notebook(os.path.join(base_dir, "notebooks", "M2_DLCV_Computer_Vision_CNN_Capstone.ipynb"))
