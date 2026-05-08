import torch.nn as nn
import os


def generate_model_inspector(model, output_file="model_inspector.html"):
    """
    Generates an interactive HTML tree to inspect any PyTorch model architecture.
    Includes parameter counts, percentages, layer metadata, and path copying.
    """

    # Get total parameters for percentage calculation
    total_params = sum(p.numel() for p in model.parameters())
    model_name = model.__class__.__name__

    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>{model_name} Inspector</title>
        <style>
            body {{ font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; background-color: #1e1e1e; color: #d4d4d4; padding: 20px; }}
            .container {{ max-width: 1300px; margin: auto; background: #252526; padding: 25px; border-radius: 8px; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }}

            /* Header Styling */
            .header-info {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #3e3e42; padding-bottom: 15px; margin-bottom: 20px; }}
            h1 {{ color: #569cd6; margin: 0; font-size: 1.5em; }}
            .total-badge {{ background: #007acc; color: white; padding: 6px 14px; border-radius: 4px; font-weight: bold; font-family: 'Consolas', monospace; }}

            /* Search Bar */
            #filterInput {{ width: 100%; padding: 12px; margin-bottom: 25px; border: 1px solid #3e3e42; background: #3c3c3c; color: white; border-radius: 4px; outline: none; box-sizing: border-box; }}
            #filterInput:focus {{ border-color: #007acc; }}

            /* Tree Structure */
            .tree ul {{ list-style: none; padding-left: 20px; border-left: 1px solid #404040; margin: 4px 0; }}
            .node-item {{ margin: 2px 0; list-style: none; }}
            .node-header {{ cursor: pointer; display: flex; align-items: center; padding: 6px 10px; border-radius: 4px; transition: background 0.1s; }}
            .node-header:hover {{ background-color: #2a2d2e; }}

            /* Icons */
            .folder-icon {{ width: 0; height: 0; margin-right: 12px; border-top: 5px solid transparent; border-bottom: 5px solid transparent; border-left: 7px solid #808080; transition: transform 0.1s; }}
            .node-item:not(.collapsed) > .node-header .folder-icon {{ transform: rotate(90deg); }}
            .leaf .folder-icon {{ border: none; width: 6px; height: 6px; background: #569cd6; border-radius: 50%; transform: none; margin-left: 4px; margin-right: 14px; }}

            /* Labels */
            .name {{ color: #ce9178; font-weight: bold; margin-right: 8px; }}
            .type {{ color: #4ec9b0; font-family: 'Consolas', monospace; font-size: 0.9em; }}
            .meta {{ color: #9cdcfe; margin-left: 15px; font-size: 0.85em; font-family: 'Consolas', monospace; font-style: italic; }}

            /* Stats and Progress Bars */
            .params-group {{ margin-left: auto; display: flex; align-items: center; gap: 15px; }}
            .params-count {{ color: #b5cea8; font-size: 0.85em; font-family: 'Consolas', monospace; min-width: 100px; text-align: right; }}
            .pct-bar-container {{ width: 80px; height: 8px; background: #333; border-radius: 4px; overflow: hidden; }}
            .pct-bar {{ height: 100%; background: #e67e22; border-radius: 4px; }}
            .pct-text {{ color: #e67e22; font-size: 0.8em; min-width: 50px; text-align: right; font-weight: bold; }}

            /* Interaction */
            .copy-path {{ opacity: 0; font-size: 10px; background: #444; color: #fff; padding: 2px 6px; border-radius: 3px; cursor: pointer; margin-left: 12px; transition: opacity 0.2s; text-transform: uppercase; }}
            .node-header:hover .copy-path {{ opacity: 1; }}

            .collapsed > ul {{ display: none; }}
            .hidden {{ display: none !important; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header-info">
                <h1>{model_name} Architecture Inspector</h1>
                <div class="total-badge">Total Params: {total_params:,}</div>
            </div>

            <input type="text" id="filterInput" placeholder="🔍 Search by layer type, dimension, or path (e.g., Linear, Conv1d, 1536)...">

            <div class="tree">
                <ul>{tree_content}</ul>
            </div>
        </div>

        <script>
            // Collapse/Expand Logic
            document.querySelectorAll('.node-header').forEach(header => {{
                header.addEventListener('click', function() {{
                    const parent = this.parentElement;
                    if (!parent.classList.contains('leaf')) parent.classList.toggle('collapsed');
                }});
            }});

            // Clipboard Integration
            function copyToClipboard(text) {{
                navigator.clipboard.writeText(text).then(() => {{
                    const originalTitle = document.title;
                    console.log("Path copied: " + text);
                }});
            }}

            // Real-time Search and Auto-Expand
            document.getElementById('filterInput').addEventListener('input', function(e) {{
                const term = e.target.value.toLowerCase();
                const allItems = document.querySelectorAll('.node-item');

                if (term === "") {{
                    allItems.forEach(item => {{
                        item.classList.remove('hidden');
                        if (!item.classList.contains('root-node')) item.classList.add('collapsed');
                    }});
                    return;
                }}

                allItems.forEach(item => {{
                    const textContent = item.innerText.toLowerCase();
                    const path = item.getAttribute('data-path').toLowerCase();

                    if (textContent.includes(term) || path.includes(term)) {{
                        item.classList.remove('hidden');
                        item.classList.remove('collapsed');

                        // Ensure all parents are visible and expanded
                        let p = item.parentElement.closest('.node-item');
                        while (p) {{ 
                            p.classList.remove('hidden', 'collapsed'); 
                            p = p.parentElement.closest('.node-item'); 
                        }}
                    }} else {{
                        item.classList.add('hidden');
                    }}
                }});
            }});
        </script>
    </body>
    </html>
    """

    def get_layer_meta(module):
        """Extracts relevant architectural parameters based on layer type."""
        meta = []
        # Linear & Embedding
        if isinstance(module, (nn.Linear, nn.Embedding)):
            in_f = module.in_features if hasattr(module, 'in_features') else getattr(module, 'num_embeddings', '?')
            out_f = module.out_features if hasattr(module, 'out_features') else getattr(module, 'embedding_dim', '?')
            meta.append(f"dim: {in_f} → {out_f}")

        # Convolutional Layers
        elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            meta.append(f"k: {module.kernel_size} | s: {module.stride}")
            meta.append(f"ch: {module.in_channels} → {module.out_channels}")

        # Normalization Layers
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
            norm_dim = getattr(module, 'normalized_shape', getattr(module, 'num_features', '?'))
            meta.append(f"norm: {norm_dim}")

        # Pooling Layers
        elif "Pool" in module.__class__.__name__:
            meta.append(f"k: {module.kernel_size} | s: {module.stride}")

        # Dropout & Activations
        elif isinstance(module, nn.Dropout):
            meta.append(f"p: {module.p}")
        elif isinstance(module, (nn.ReLU, nn.GELU, nn.SiLU, nn.Softmax)):
            meta.append("activation")

        return " | ".join(meta)

    def build_tree(module, name, path="", is_root=False):
        children = list(module.named_children())
        # Recursive parameter count for current scope
        node_params = sum(p.numel() for p in module.parameters())
        class_name = module.__class__.__name__
        is_folder = len(children) > 0

        meta_info = get_layer_meta(module)
        pct = (node_params / total_params) * 100 if total_params > 0 else 0
        current_path = name if is_root else f"{path}.{name}"

        item_classes = ["node-item"]
        if is_root: item_classes.append("root-node")
        if is_folder and not is_root: item_classes.append("collapsed")
        if not is_folder: item_classes.append("leaf")

        html = f'<li class="{" ".join(item_classes)}" data-path="{current_path}">'
        html += f'<div class="node-header">'
        html += f'<span class="folder-icon"></span>'
        html += f'<span class="name">{name}</span>'
        html += f'<span class="type">[{class_name}]</span>'

        if meta_info:
            html += f'<span class="meta">{meta_info}</span>'

        html += f'<span class="copy-path" onclick="event.stopPropagation(); copyToClipboard(\'{current_path}\')">copy path</span>'

        html += f'<div class="params-group">'
        html += f'<span class="params-count">{node_params:,}</span>'
        html += f'<div class="pct-bar-container"><div class="pct-bar" style="width: {pct}%"></div></div>'
        html += f'<span class="pct-text">{pct:.2f}%</span>'
        html += '</div></div>'

        if is_folder:
            html += '<ul>'
            for child_name, child_module in children:
                html += build_tree(child_module, child_name, current_path)
            html += '</ul>'

        html += '</li>'
        return html

    print(f"Inspecting {model_name} architecture...")
    tree_content = build_tree(model, "model", is_root=True)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_template.format(
            model_name=model_name,
            tree_content=tree_content,
            total_params=total_params
        ))

    print(f"Generated: {os.path.abspath(output_file)}")
