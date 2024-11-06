"""
This is a streamlit notebook. Run it with

```bash
streamlit run explore_sample_data_streamlit.py
```
"""
import base64
import dataclasses
import io
import json
import random
from dataclasses import dataclass
from typing import Optional

import jax
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from IPython.display import DisplayObject

import apebench

st.set_page_config(layout="wide")

with st.sidebar:
    st.title("APEBench Scenario Overview")

    scenario = st.selectbox(
        "Select Scenario",
        list(apebench.scenarios.scenario_dict.keys()),
        index=0,
    )

    dimension_type = st.select_slider(
        "Number of Spatial Dimensions (ST=Spatio-Temporal plot)",
        options=["1d ST", "1d", "2d", "2d ST", "3d"],
    )

    dataset = st.select_slider(
        "Dataset",
        options=["Train", "Test"],
        value="Test",
    )

    mode = st.select_slider(
        "Mode",
        options=["Ref", "Coarse"],
    )

    trajectory_index = st.slider(
        "Trajectory Index",
        min_value=0,
        max_value=20,
        value=0,
    )


if dimension_type in ["1d ST", "1d"]:
    num_spatial_dims = 1
elif dimension_type in ["2d ST", "2d"]:
    num_spatial_dims = 2
elif dimension_type == "3d":
    num_spatial_dims = 3

if num_spatial_dims == 3:
    st.warning("Reducing to 32 num points for 3D visualization")
    scenario = apebench.scenarios.scenario_dict[scenario](
        num_spatial_dims=num_spatial_dims, num_points=32
    )
else:
    scenario = apebench.scenarios.scenario_dict[scenario](
        num_spatial_dims=num_spatial_dims
    )

if dataset == "Train":
    if mode == "Ref":
        trj_set = scenario.get_train_data()
    else:
        trj_set = scenario.get_train_data_coarse()
elif dataset == "Test":
    if mode == "Ref":
        trj_set = scenario.get_test_data()
    else:
        trj_set = scenario.get_test_data_coarse()

trj = trj_set[trajectory_index]

TEMPLATE_IFRAME = """
    <div>
        <iframe id="{canvas_id}" src="https://vape.niedermayr.dev/?inline" width="{canvas_width}" height="{canvas_height}" frameBorder="0" sandbox="allow-same-origin allow-scripts"></iframe>
    </div>
    <script>

        window.addEventListener(
            "message",
            (event) => {{
                if (event.data !== "ready") {{
                    return;
                }}
                let data_decoded = Uint8Array.from(atob("{data_code}"), c => c.charCodeAt(0));
                let cmap_decoded = Uint8Array.from(atob("{cmap_code}"), c => c.charCodeAt(0));
                const iframe = document.getElementById("{canvas_id}");
                if (iframe === null) return;
                iframe.contentWindow.postMessage({{
                    volume: data_decoded,
                    cmap: cmap_decoded,
                    settings: {settings_json}
                }},
                "*");
            }},
            false,
        );
    </script>
"""


@dataclass(unsafe_hash=True)
class ViewerSettings:
    width: int
    height: int
    background_color: tuple
    show_colormap_editor: bool
    show_volume_info: bool
    vmin: Optional[float]
    vmax: Optional[float]
    distance_scale: float


def show(
    data: np.ndarray,
    colormap,
    width: int = 800,
    height: int = 600,
    background_color=(0.0, 0.0, 0.0, 1.0),
    show_colormap_editor=False,
    show_volume_info=False,
    vmin=None,
    vmax=None,
    distance_scale=1.0,
):
    return VolumeRenderer(
        data,
        colormap,
        ViewerSettings(
            width,
            height,
            background_color,
            show_colormap_editor,
            show_volume_info,
            vmin,
            vmax,
            distance_scale,
        ),
    )


class VolumeRenderer(DisplayObject):
    def __init__(self, data: np.ndarray, colormap, settings: ViewerSettings):
        super(VolumeRenderer, self).__init__(
            data={"volume": data, "cmap": colormap, "settings": settings}
        )

    def _repr_html_(self):
        data = self.data["volume"]
        colormap = self.data["cmap"]
        settings = self.data["settings"]
        buffer = io.BytesIO()
        np.save(buffer, data.astype(np.float32))
        data_code = base64.b64encode(buffer.getvalue())

        buffer2 = io.BytesIO()
        colormap_data = colormap(np.linspace(0, 1, 256)).astype(np.float32)
        np.save(buffer2, colormap_data)
        cmap_code = base64.b64encode(buffer2.getvalue())

        canvas_id = f"v4dv_canvas_{str(random.randint(0, 2**32))}"
        html_code = TEMPLATE_IFRAME.format(
            canvas_id=canvas_id,
            data_code=data_code.decode("utf-8"),
            cmap_code=cmap_code.decode("utf-8"),
            canvas_width=settings.width,
            canvas_height=settings.height,
            settings_json=json.dumps(dataclasses.asdict(settings)),
        )
        return html_code

    def __html__(self):
        """
        This method exists to inform other HTML-using modules (e.g. Markupsafe,
        htmltag, etc) that this object is HTML and does not need things like
        special characters (<>&) escaped.
        """
        return self._repr_html_()


if dimension_type == "1d ST":
    fig = apebench.exponax.viz.plot_spatio_temporal(trj, vlim=scenario.vlim)
    st.pyplot(fig)
elif dimension_type == "1d":
    ani = apebench.exponax.viz.animate_state_1d(trj, vlim=scenario.vlim)
    components.html(ani.to_jshtml(), height=800)
elif dimension_type == "2d":
    ani = apebench.exponax.viz.animate_state_2d(trj, vlim=scenario.vlim)
    components.html(ani.to_jshtml(), height=800)
elif dimension_type == "2d ST":
    trj_wrapped = jax.vmap(apebench.exponax.wrap_bc)(trj)
    trj_rearranged = trj_wrapped.transpose(1, 0, 2, 3)[None]
    components.html(
        show(
            trj_rearranged,
            plt.get_cmap("RdBu_r"),
            width=1500,
            height=800,
            show_colormap_editor=True,
            show_volume_info=True,
        ).__html__(),
        height=800,
    )
elif dimension_type == "3d":
    trj_wrapped = jax.vmap(apebench.exponax.wrap_bc)(trj)
    components.html(
        show(
            trj_wrapped,
            plt.get_cmap("RdBu_r"),
            width=1500,
            height=800,
            show_colormap_editor=True,
            show_volume_info=True,
        ).__html__(),
        height=800,
    )
