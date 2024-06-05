use miette::{IntoDiagnostic, Result};
use wgsl_bindgen::{
    GlamWgslTypeMap, WgslBindgenOptionBuilder, WgslShaderSourceType, WgslTypeSerializeStrategy,
};

fn main() -> Result<()> {
    WgslBindgenOptionBuilder::default()
        .serialization_strategy(WgslTypeSerializeStrategy::Bytemuck)
        .emit_rerun_if_change(true)
        .short_constructor(0)
        .type_map(GlamWgslTypeMap)
        .workspace_root("src/shaders")
        .add_entry_point("src/shaders/prefix_sum_add_scanned_sums.wgsl")
        .add_entry_point("src/shaders/prefix_sum_scan_sums.wgsl")
        .add_entry_point("src/shaders/prefix_sum_scan.wgsl")
        .output("src/shaders/mod.rs")
        .shader_source_type(WgslShaderSourceType::UseComposerEmbed)
        .build()?
        .generate()
        .into_diagnostic()
}
