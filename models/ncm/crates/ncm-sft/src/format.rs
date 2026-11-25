//! The NCM chat format at text level; the renderer is shared in `ncm_model::chat`.

pub use ncm_model::chat::{
    IM_END, IM_START, RESERVED, Rendered, THINK_CLOSE, THINK_OPEN, TOOL_CALL_CLOSE, TOOL_CALL_OPEN,
    TOOL_RESPONSE_CLOSE, TOOL_RESPONSE_OPEN, TOOLS_CLOSE, TOOLS_OPEN, render, render_tools_block,
};
