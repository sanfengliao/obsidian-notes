# Demo代码

假设窗口宽度是 1000 x 800

```HTML
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Document</title>
  <style>
    * { 
      margin: 0;
      padding: 0;
    }
    html {
       margin-left: 1px;
       margin-top: 1px;
    },
    .box {
      width: 100px;
      height: 100px;
      background-color: red;
    }
  </style>
</head>
<body>
  <div class="box"></div>
</body>
</html>
```

# Document BlockLayout

## 初始化PreviousInflowPosition

```C++
PreviousInflowPosition previous_inflow_position = {
      LayoutUnit(), constraint_space.GetMarginStrut(),
      is_resuming_ ? LayoutUnit() : container_builder_.Padding().block_start,
      /* self_collapsing_child_had_clearance */ false};
/*
  previous_inflow_position:  {
    // 上一个child相对于parent结束的位置，一开始是paddingTop
    logical_block_offset: 0,
    margin_strut: {
        positive_margin: 0,
        negative_margin: 0,
    },
    block_end_annotation_space: 0,
    self_collapsing_child_had_clearance: 0,
}
*/
```

## ResolveBfcBlockOffset

```Java
  
  if (content_edge || is_resuming_ ||
      constraint_space.IsNewFormattingContext()) {
    bool discard_subsequent_margins =
        previous_inflow_position.margin_strut.discard_margins && !content_edge; // false
    if (!ResolveBfcBlockOffset(&previous_inflow_position)) {
      return container_builder_.Abort(LayoutResult::kBfcBlockOffsetResolved);
    }
    // 设置成paddingTop的值
    previous_inflow_position.logical_block_offset = content_edge;

    // If we resolved the BFC block offset now, the margin strut has been
    // reset. If margins are to be discarded, and this box would otherwise have
    // adjoining margins between its own margin and those subsequent content,
    // we need to make sure subsequent content discard theirs.
    if (discard_subsequent_margins)
      previous_inflow_position.margin_strut.discard_margins = true;
  }
  
  // container_builder_.BfcBlockOffset():  0(LayoutUnit)
  // constraint_space.GetMarginStrut().IsEmpty(): true 
```

## HandleNewFormattingContext

```C++
 // child = html, 
 status = HandleNewFormattingContext(
            child, To<BlockBreakToken>(child_break_token),
            &previous_inflow_position);
```

### ComputeChildData

```C++
InflowChildData child_data =
      ComputeChildData(*previous_inflow_position, child, child_break_token,
                       /* is_new_fc */ true)
InflowChildData BlockLayoutAlgorithm::ComputeChildData(
    const PreviousInflowPosition& previous_inflow_position,
    LayoutInputNode child,
    const BreakToken* child_break_token,
    bool is_new_fc) {

  // Calculate margins in parent's writing mode.
  LayoutUnit additional_line_offset;
  BoxStrut margins =
      CalculateMargins(child, is_new_fc, &additional_line_offset);

  MarginStrut margin_strut = previous_inflow_position.margin_strut;

  LayoutUnit logical_block_offset =
      previous_inflow_position.logical_block_offset;

  margin_strut.Append(margins.block_start,
                      child.Style().HasMarginBlockStartQuirk());


  TextDirection direction = GetConstraintSpace().Direction();
  BfcOffset child_bfc_offset = {
      //bfc line Offset = 父元素bfcOffset + padding + 子元素的marginLeft
      GetConstraintSpace().GetBfcOffset().line_offset +
          BorderScrollbarPadding().LineLeft(direction) +
          additional_line_offset + margins.LineLeft(direction),
      // bfc block offset = 父元素的bfc block offset + 上一个元素结束的位置(相对于父元素) 
      // 这里还没有加上margin，因为要处理margin合并的问题
      // 如果父元素创建新bfc，那么BfcBlockOffset() 返回的是0
      BfcBlockOffset() + logical_block_offset};

  return InflowChildData(child_bfc_offset, margin_strut, margins);
}

/* 
child_data = {
    margins: BoxStrut{
        inline_start: 1,
        inline_end: 0,
        block_start: 1,
        block_end: 0,
    },
    margin_strut: MarginStrut{
        positive_margin: 1,
        negative_margin: 0,
    },
    bfc_offset_estimate: BfcOffset{
        lineOffset: 1,
        block_offset: 0
    }
} 
*/
```

### 设置一下临时变量

```C++

// 不包含child marginLeft时候的偏移，
// 因为margin是auto的时候，无法立刻计算的实际的偏移
LayoutUnit child_origin_line_offset =
      constraint_space.GetBfcOffset().line_offset +
      BorderScrollbarPadding().LineLeft(direction); 
MarginStrut adjoining_margin_strut(previous_inflow_position->margin_strut);

// margin合并
adjoining_margin_strut.Append(child_data.margins.block_start,
                                child_style.HasMarginBlockStartQuirk()); 
// 加上margin合并之后的值，最终是child的bfc block offset
LayoutUnit adjoining_bfc_offset_estimate =
      child_data.bfc_offset_estimate.block_offset +
      adjoining_margin_strut.Sum(); // 1

LayoutUnit non_adjoining_bfc_offset_estimate =
      child_data.bfc_offset_estimate.block_offset +
      previous_inflow_position->margin_strut.Sum(); // 0

// 等于上一个child结束的位置 + 他们之间的margin。
// 如果第一个，那么就是parent的bfcOffset + parent的paddingTop + child的marginTop
LayoutUnit child_bfc_offset_estimate = adjoining_bfc_offset_estimate; // 1
bool bfc_offset_already_resolved = false;
bool child_determined_bfc_offset = false;
bool child_margin_got_separated = false;
bool has_adjoining_floats = false;

if (!container_builder_.BfcBlockOffset()) {
  // ...
}


bool abort_if_cleared = child_data.margins.block_start != LayoutUnit() &&
                       !child_margin_got_separated &&
                         child_determined_bfc_offset; // false
```

### LayoutNewFormattingContext

```C++


const LayoutResult* layout_result = LayoutNewFormattingContext(
  child, child_break_token, child_data,
  {child_origin_line_offset, child_bfc_offset_estimate}, abort_if_cleared,
  &child_bfc_offset, &resolved_margins)

// origin_offset = { child_origin_line_offset, child_bfc_offset_estimate}
const LayoutResult* BlockLayoutAlgorithm::LayoutNewFormattingContext(
    LayoutInputNode child,
    const BlockBreakToken* child_break_token,
    const InflowChildData& child_data,
    BfcOffset origin_offset, // {line_offset: 0, block_offest: 1}
    bool abort_if_cleared,
    BfcOffset* out_child_bfc_offset,
    BoxStrut* out_resolved_margins) {

}
```

1. #### 初始化`opportunities`
    

```C++
  LayoutOpportunityVector opportunities =
      GetExclusionSpace().AllLayoutOpportunities(
          origin_offset, ChildAvailableSize().inline_size); // ChildAvailableSize(): LogicalSize{inline_size: 1000, block_size: 800 } 

 /* opportunities = [
     LayoutOpportunity{
         rect: BfcRect{
             start_offset: BfcOffset{ line_offset: 0,  block_offset: 1}
             end_offset: BfcOffset{ line_offset: available_inline_size + start_offset.line_offset,  block_offset: Layout:max()}
         }
     }
 ]
 
 */
```

2. #### 遍历`opportunities`开始布局
    

```C++
for (const auto& opportunity : opportunities) {}
```

3. #### 初始化各种变量
    

```Java
bool has_floats_on_line_left =
    opportunity.rect.LineStartOffset() != origin_offset.line_offset; // false
bool has_floats_on_line_right =
    opportunity.rect.LineEndOffset() !=
    (origin_offset.line_offset + ChildAvailableSize().inline_size); // false
bool can_expand_outside_opportunity =
    !has_floats_on_line_left && !has_floats_on_line_right; // false

const LayoutUnit line_left_margin = child_data.margins.LineLeft(direction); // 1
const LayoutUnit line_right_margin =
    child_data.margins.LineRight(direction); // 0


LayoutUnit line_left_offset = opportunity.rect.LineStartOffset(); // 0
LayoutUnit line_right_offset = opportunity.rect.LineEndOffset(); // 1000

if (can_expand_outside_opportunity) {
  line_left_offset += line_left_margin;
  line_right_offset -= line_right_margin;
} else {
  // 1
  line_left_offset = std::max(
      line_left_offset,
      origin_offset.line_offset + line_left_margin.ClampNegativeToZero()); 
  // 1000
  line_right_offset = std::min(line_right_offset,
                               origin_offset.line_offset +
                                   ChildAvailableSize().inline_size -
                                   line_right_margin.ClampNegativeToZero()); 
}
LayoutUnit opportunity_size =
    (line_right_offset - line_left_offset).ClampNegativeToZero(); // 999
LayoutUnit child_available_inline_size =
    (opportunity_size + child_data.margins.InlineSum())
        .ClampNegativeToZero(); // 1000
   
```

4. #### 创建html元素的`ConstraintSpace`并布局
    

```C++
ConstraintSpace child_space = CreateConstraintSpaceForChild(
    child, child_break_token, child_data,
    {child_available_inline_size, ChildAvailableSize().block_size},
    /* is_new_fc */ true, opportunity.rect.start_offset.block_offset);

// All formatting context roots (like this child) should start with an empty
// exclusion space.
DCHECK(child_space.GetExclusionSpace().IsEmpty());

const LayoutResult* layout_result = LayoutBlockChild(
    child_space, child_break_token, early_break_,
    /* column_spanner_path */ nullptr, &To<BlockNode>(child));5
```

5. #### 处理`margin`是`auto`的情况
    

```C++
BoxStrut auto_margins = child_data.margins;
LayoutUnit text_align_offset;
bool has_auto_margins = false;
if (child.IsListMarker()) {
  // ...
} else {
  if (child_style.MarginInlineStartUsing(style).IsAuto() ||
      child_style.MarginInlineEndUsing(style).IsAuto()) {
    has_auto_margins = true;
    ResolveInlineAutoMargins(child_style, style,
                             child_available_inline_size,
                             fragment.InlineSize(), &auto_margins);
  } else {
    // Handle -webkit- values for text-align.
    text_align_offset = WebkitTextAlignAndJustifySelfOffset(
        child_style, style, opportunity.rect.InlineSize(),
        child_data.margins, [&]() { return fragment.InlineSize(); });
  }
}
/* 

*/
```

在这个demo下, html的`auto_margins`是:

```JSON
auto_margins: BoxStrut{
    inline_start: 1,
    inline_end: 0,
    block_start: 1,
    block_end: 0
}
```

如果`html` 设置`width: 500px, margin: 1 auto`，那么

```JSON
auto_margins: BoxStrut{
    inline_start: 250,
    inline_end: 250,
    block_start: 1,
    block_end: 0
}
```

6. #### 计算`child_bfc_offset`
    

主要计算margin时auto时，子元素的bfc line offset。

```C++
BfcOffset child_bfc_offset = {LayoutUnit(),
                              opportunity.rect.BlockStartOffset()};
// 计算margin是auto时
if (direction == TextDirection::kLtr) {
  // 当margin是auto时, line_left_margin为0， auto_margin_line_left为最终确定的margin值
  // 当margin时具体数值时, auto_margin_line_left时0 ，这是符合预期的，因为前面计算line_left_offset加上line_left_margin了
  LayoutUnit auto_margin_line_left =
      auto_margins.LineLeft(direction) - line_left_margin;
  child_bfc_offset.line_offset =
      line_left_offset + auto_margin_line_left + text_align_offset;
} else {
  LayoutUnit auto_margin_line_right =
      auto_margins.LineRight(direction) - line_right_margin;
  child_bfc_offset.line_offset = line_right_offset - text_align_offset -
                                 auto_margin_line_right -
                                 fragment.InlineSize();
}
```

7. #### 计算resolved_margins
    

处理margin是auto的时候，最终的值，这里为什么要再算一次，暂时不清楚

```C++
// auto-margins are "fun". To ensure round tripping from getComputedStyle
// the used values are relative to the content-box edge, rather than the
// opportunity edge.
BoxStrut resolved_margins = child_data.margins;
if (has_auto_margins) {
  LayoutUnit inline_offset =
      LogicalFromBfcLineOffset(child_bfc_offset.line_offset,
                               container_builder_.BfcLineOffset(),
                               fragment.InlineSize(),
                               container_builder_.InlineSize(), direction) -
      BorderScrollbarPadding().inline_start;
  if (child_style.MarginInlineStartUsing(style).IsAuto()) {
    resolved_margins.inline_start = inline_offset;
  }
  if (child_style.MarginInlineEndUsing(style).IsAuto()) {
    resolved_margins.inline_end = ChildAvailableSize().inline_size -
                                  inline_offset - fragment.InlineSize();
  }
}
```

1. 返回结果
    

```C++
*out_child_bfc_offset = child_bfc_offset;
*out_resolved_margins = resolved_margins;
return layout_result;
```

  

### LogicalFromBfcOffsets

计算child相对于parent的偏移量（已经算上了child的marginTop)

```C++
LogicalOffset logical_offset = LogicalFromBfcOffsets(
      child_bfc_offset, ContainerBfcOffset(), fragment.InlineSize(),
      container_builder_.InlineSize(), constraint_space.Direction());

LogicalOffset LogicalFromBfcOffsets(const BfcOffset& child_bfc_offset,
                                    const BfcOffset& parent_bfc_offset,
                                    LayoutUnit child_inline_size,
                                    LayoutUnit parent_inline_size,
                                    TextDirection direction) {
  // 子元素在父元素的偏移 = 子元素的bfcOffset - 父元素的bfcOffset 
  // 已经算上了child的marginTop
  
  // 水平方向的偏移
  LayoutUnit inline_offset = LogicalFromBfcLineOffset(
      child_bfc_offset.line_offset, parent_bfc_offset.line_offset,
      child_inline_size, parent_inline_size, direction);

  return {inline_offset,
          child_bfc_offset.block_offset - parent_bfc_offset.block_offset};
}
      
LayoutUnit LogicalFromBfcLineOffset(LayoutUnit child_bfc_line_offset,
                                    LayoutUnit parent_bfc_line_offset,
                                    LayoutUnit child_inline_size,
                                    LayoutUnit parent_inline_size,
                                    TextDirection direction) {

  // 
  LayoutUnit relative_line_offset =
      child_bfc_line_offset - parent_bfc_line_offset;

  LayoutUnit inline_offset =
      direction == TextDirection::kLtr
          ? relative_line_offset
          : parent_inline_size - relative_line_offset - child_inline_size;

  return inline_offset;
}


```

  
### ComputeInflowPosition

更新previous_inflow_position，用来给一个child布局

```C++
if (!child_break_token || !child_break_token->IsInParallelFlow()) {
    *previous_inflow_position = ComputeInflowPosition(
        *previous_inflow_position, child, child_data,
        child_bfc_offset.block_offset, logical_offset, *layout_result, fragment,
        /* self_collapsing_child_had_clearance */ false);
}

PreviousInflowPosition BlockLayoutAlgorithm::ComputeInflowPosition(
    const PreviousInflowPosition& previous_inflow_position,
    const LayoutInputNode child,
    const InflowChildData& child_data,
    const std::optional<LayoutUnit>& child_bfc_block_offset,
    // 子元素相对于父元素的偏移
    const LogicalOffset& logical_offset,
    const LayoutResult& layout_result,
    const LogicalFragment& fragment,
    bool self_collapsing_child_had_clearance) {
  // Determine the child's end logical offset, for the next child to use.
  LayoutUnit logical_block_offset;
  std::optional<LayoutUnit> clearance_after_line;
  std::optional<LayoutUnit> trim_block_end_by;

  const bool is_self_collapsing = layout_result.IsSelfCollapsing();
  if (is_self_collapsing) {
    // ...
  } else {
    // 下一个child的开始位置(还没有加上下一个child marginTop) = 
    // 上一个child相对父元素的开始位置(已经处理好了marginTop, 因为这个值是通过bfcOffset计算得到的)
    // + 上一个child的block_size
    logical_block_offset = logical_offset.block_offset + fragment.BlockSize();
  }

  MarginStrut margin_strut = layout_result.EndMarginStrut();
  // 处理child和他child的最后一个child的margin合并
  margin_strut.Append(child_data.margins.block_end, is_quirky);

  return {logical_block_offset, margin_strut, annotation_space,
          self_or_sibling_self_collapsing_child_had_clearance};
}
```
