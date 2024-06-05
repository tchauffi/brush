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
        .workspace_root("src/splat_render/shaders")
        .add_entry_point("src/splat_render/shaders/project_forward.wgsl")
        .add_entry_point("src/splat_render/shaders/map_gaussian_to_intersects.wgsl")
        .add_entry_point("src/splat_render/shaders/get_tile_bin_edges.wgsl")
        .add_entry_point("src/splat_render/shaders/rasterize.wgsl")
        .add_entry_point("src/splat_render/shaders/rasterize_backwards.wgsl")
        .add_entry_point("src/splat_render/shaders/project_backwards.wgsl")
        .output("src/splat_render/shaders/mod.rs")
        .shader_source_type(WgslShaderSourceType::UseComposerEmbed)
        .build()?
        .generate()
        .into_diagnostic()
}
