# DESIGN.md

# LS Manufacturing AI Dashboard Design System

This document defines the mandatory UI/UX and visual design rules for this project.

All frontend implementation and redesign work MUST follow this document.

This project is a desktop-first enterprise manufacturing AI dashboard.
It must NOT look like a mobile application, tablet application, consumer SaaS landing page,
or generic Bootstrap admin template.

---

# 1. Technology Stack

This project uses Plotly Dash.

Do NOT migrate the frontend to:
- React
- Next.js
- Streamlit
- Vue
- Svelte
- another frontend framework

Required UI stack:

- Plotly Dash
- dash-mantine-components
- Dash AG Grid
- Plotly Graph
- dash-iconify

Use Dash callbacks for interactions.

---

# 2. Design Goal

The interface should feel like a production-grade enterprise dashboard
used by manufacturing, supply-chain, production-planning,
and inventory-management teams.

The visual direction is:

- Modern
- Professional
- Industrial
- Enterprise-oriented
- Clean
- Dense but readable
- Data-first
- Operational rather than decorative

The dashboard should communicate:

1. Current operational status
2. AI forecast
3. Difference between plan and forecast
4. Abnormal or actionable items
5. Recommended actions

The design must prioritize decision-making over decoration.

Because the final project is demonstrated directly in Dash without a separate
slide presentation, the application must also expose concise evaluation evidence:

- Project purpose and business questions
- Dataset roles and KAMP sources
- Validation design and leakage prevention
- Baseline and candidate model comparison
- Data quality evidence
- Key conclusions, limitations, future improvements, and team contributions

These views must remain dashboard-like and interactive. They must NOT imitate
a presentation deck with oversized titles, long prose blocks, or slide-by-slide layouts.

---

# 3. Desktop Canvas & Layout

## Primary Design Target

The primary design viewport is:

| Item | Value |
|---|---:|
| Design viewport width | 1920px |
| Fixed left sidebar | 240px |
| Main workspace width at 1920px | 1680px |
| Main content max-width | 1680px |
| Main page padding | 24px |
| Card gap | 16px - 20px |

At a 1920px viewport:

1920 - 240 = 1680px

Therefore:

240px fixed left sidebar
+
1680px main workspace

must create the primary desktop composition.

The previous 240px + 1440px + 240px centered composition is no longer the
primary application layout. The 1440px centered composition may still be used
inside standalone reports or previews, but not for the integrated application shell.

Recommended implementation:

```css
.app-shell {
    min-height: 100vh;
    display: grid;
    grid-template-columns: 240px minmax(0, 1fr);
}

.app-sidebar {
    position: sticky;
    top: 0;
    height: 100vh;
}

.app-main {
    min-width: 0;
}

.dashboard-container {
    width: 100%;
    max-width: 1680px;
    margin: 0 auto;
    padding: 24px;
}
```

The 1920px value is the PRIMARY DESIGN CANVAS.

Do not unnecessarily set:

```css
body {
    width: 1920px;
}
```

because that creates artificial horizontal scrolling.

Instead, design and visually validate the interface
at a 1920x1080 desktop viewport.

## 3.1 Application Shell

The integrated application must use one persistent desktop App Shell.

The shell contains:

1. Fixed left sidebar
2. Main workspace
3. Optional compact top utility row inside the main workspace
4. Page title and context
5. Secondary functional tabs
6. Shared context filters
7. Active tab content

The previous centered top global navigation must be removed when the sidebar
shell is implemented. A compact utility row may retain breadcrumb, data status,
last updated time, notifications, and user information, but it must not duplicate
the four primary navigation items.

## 3.2 Primary Sidebar Navigation

The sidebar is the only primary navigation for the integrated application.

Required primary items, in this exact order:

1. 통합 현황
2. 공급망 예측
3. 예지보전
4. 품질 보증

Rules:

- Use `공급망 예측` as the primary navigation label even when the page title is
  the more specific `D+3 발주량 예측`.
- Show one clearly active primary item at all times.
- Use LS Blue and the established LS color tokens for active and focus states.
- Keep item height compact and suitable for an enterprise desktop application.
- Do not use floating navigation, oversized icons, or mobile-style pill buttons.
- Icons may be used at 18px - 20px when they materially improve scanning.
- The sidebar must remain visible at the supported desktop resolutions.
- The brand lockup belongs at the top of the sidebar.

## 3.3 AI Copilot Placement

AI Copilot must appear below the four primary navigation items and remain
anchored near the bottom of the sidebar.

The layout order is:

Brand
↓
Primary navigation
↓
Flexible spacer
↓
AI Copilot entry

The AI Copilot entry must:

- Be visually separated from primary navigation
- Use a compact action-card or button treatment
- Include a short label and optional one-line helper text
- Not appear as a fifth business domain
- Open a right-side Drawer when Copilot interaction is implemented
- Not imply that AI answers are available until the underlying feature is connected

## 3.4 Secondary Functional Tabs

Secondary tabs organize the functions inside the selected primary business domain.
They must appear below the page title or breadcrumb and above shared filters and content.

Use `dmc.Tabs` with compact desktop proportions.

Required behavior:

- Recommended tab height: 40px - 44px
- Use LS Blue for the active tab
- Prefer an underline, restrained filled state, or border treatment
- Do not use large pill-shaped mobile tabs
- Keep labels on one line at the primary desktop resolutions
- Preserve relevant filter and selection state when switching tabs
- Support browser history or URL-addressable tab state where practical
- Do not turn one-off actions, Drawers, or Modals into tabs

Required information architecture:

| Primary domain | Required secondary tabs |
|---|---|
| 통합 현황 | 통합 요약 · 프로젝트 개요 · 데이터 품질 · 결과 조회 · 결론·한계 |
| 공급망 예측 | 예측 분석 · 검토 목록 · 모델 검증 |
| 예지보전 | 이상 분석 · 이벤트 검토 · 모델 검증 |
| 품질 보증 | 품질 분석 · 판정·조치 · 모델 검증 |

Do not add empty tabs merely to make every domain have the same number of tabs.

## 3.5 Shared Context and Tab-Local Controls

Controls that affect multiple tabs must remain visible and retain their state.

Shared context by domain:

- 공급망 예측: 기준일, 부품, 예측 모델
- 예지보전: 시험 파일, 지도 모델, 비지도 모델
- 품질 보증: 시험 ID, 공정, 분석 진행률, 운영 모델

Tab-local controls:

- 공급망 `검토 목록`: 부품 검색, 상향·하향 검토 필터, 목록 내보내기
- 예지보전 `이벤트 검토`: 이벤트 상태 필터, 상세 보기, 확인 처리, 이벤트 내보내기
- 품질 보증 `판정·조치`: 작업자 판정, 메모, 결과 내보내기

The following remain actions or contextual surfaces, not secondary tabs:

- CSV 재예측
- 모델 정보
- 데이터 품질 detail Modal or Drawer opened from a track page
- Part detail
- Event detail
- Export and download actions

## 3.6 Navigation Boundaries

The three current track applications may remain separate Dash modules during
implementation, but the final integrated experience must provide working navigation
between them and the integrated overview.

The final application must avoid port or URL collisions and must define one stable
destination for each primary sidebar item. Visual navigation that does not actually
open the selected domain is not complete.

# 4. Desktop-First Rule

This is NOT a mobile-first application.

The dashboard should visually use the horizontal space available
on a desktop monitor.

Primary validation resolution:

1920 x 1080

Secondary desktop validation resolutions:

1600 x 900
1440 x 900
1366 x 768

The interface may adapt to smaller desktop screens,
but must preserve the desktop dashboard composition.

Do NOT redesign the page into a mobile layout.

# 5. Forbidden Layout Patterns

The following are prohibited.

DO NOT
Create a narrow centered layout that resembles a mobile application
Design the dashboard at tablet proportions
Design the dashboard at mobile proportions
Collapse the entire dashboard into a single vertical column
Make cards unnecessarily tall
Use oversized hero sections
Use oversized headings
Use oversized buttons
Use oversized KPI numbers
Use excessive whitespace
Create extremely narrow cards
Create large empty decorative panels
Stack every filter vertically
Make the dashboard look like a landing page
Make the dashboard look like a smartphone application
Use floating mobile-style navigation
Use large pill-shaped mobile buttons everywhere
Use giant rounded cards

The page should make efficient use of the full main workspace remaining after
the fixed sidebar. At 1920px this is a 1680px workspace with 24px page padding.

# 6. LS Brand Color System

The color system is based on the official LS CI guidelines.

Official LS CI reference:

https://www.lsholdings.com/ko/media/ci

Primary Brand Colors
LS Blue

Official RGB:

RGB(10, 30, 90)

HEX:

#0A1E5A

Use LS Blue as the primary enterprise color.

Recommended usage:

Primary navigation
Active tabs
Main buttons
Main AI forecast line
Important section titles
Selected states
Primary icons
Focus states

Do NOT fill huge portions of the dashboard with LS Blue.

LS Red

Official RGB:

RGB(250, 0, 45)

HEX:

#FA002D

Use LS Red carefully as a point / alert color.

Recommended usage:

Critical warning
Significant negative deviation
High-risk item
Exception requiring immediate review
Small brand accent
Critical status badge

LS Red MUST NOT be used as the default color
for large cards, backgrounds, tables, or charts.

Red should remain visually powerful because it is used sparingly.

# 7. LS Supporting Colors

Official LS secondary colors may be used when appropriate.

LS Green
RGB(0, 155, 180)
HEX #009BB4

Recommended usage:

Positive operational state
Stable status
Secondary chart series
LS Light Blue
RGB(5, 105, 160)
HEX #0569A0

Recommended usage:

Secondary data series
Supporting visualizations
Hover states
Information states
LS Gray
RGB(125, 130, 130)
HEX #7D8282

Recommended usage:

Secondary text
Muted labels
Non-critical chart series
Disabled information
# 8. Application Color Tokens

Do NOT manually scatter raw color values throughout Python files.

Define reusable CSS variables inside:

assets/theme.css

Recommended variables:

:root {
    /* LS Brand */
    --ls-blue: #0A1E5A;
    --ls-red: #FA002D;
    --ls-green: #009BB4;
    --ls-light-blue: #0569A0;
    --ls-gray: #7D8282;

    /* Neutral surfaces */
    --background: #F5F7FA;
    --surface: #FFFFFF;
    --surface-secondary: #F8F9FB;

    /* Text */
    --text-primary: #172033;
    --text-secondary: #667085;
    --text-muted: #98A2B3;

    /* Border */
    --border-light: #E7EAF0;
    --border-medium: #D8DDE6;

    /* Status */
    --success: #009BB4;
    --warning: #D98B00;
    --danger: #FA002D;
    --info: #0569A0;

    /* LS tinted backgrounds */
    --ls-blue-soft: #F0F3FA;
    --ls-red-soft: #FFF1F3;
    --ls-green-soft: #EDF9FA;

    /* Application shell */
    --sidebar-width: 240px;
    --sidebar-width-compact: 216px;
    --sidebar-bg: var(--ls-blue);
    --sidebar-text: rgba(255, 255, 255, 0.78);
    --sidebar-text-active: #FFFFFF;
    --sidebar-active-bg: rgba(255, 255, 255, 0.12);
    --sidebar-border: rgba(255, 255, 255, 0.14);
}

Neutral colors may be adjusted slightly for readability,
but LS Blue and LS Red must retain their official RGB values.

# 9. Color Usage Philosophy

The application must NOT look colorful for the sake of being colorful.

Use approximately:

70-80% neutral colors
15-20% LS Blue family
< 5% LS Red / alert colors

The background should remain visually quiet.

Recommended hierarchy:

Background
    ↓
Light neutral gray

Cards
    ↓
White

Primary information
    ↓
Dark neutral

Brand / AI
    ↓
LS Blue

Exceptions
    ↓
LS Red

Positive / stable
    ↓
LS Green

Avoid rainbow dashboards.

# 10. Typography

Project identity (user-approved): **BatteryFlow AI 운영센터**.
Use the provided `ci_img02.png` on the login page and `ci_img20.png` in the sidebar,
without recoloring or distortion. Do not add a white background behind the sidebar logo.
Use `font-family: 'noto_d', sans-serif` across the login page, Dash/Mantine UI,
AG Grid tables, and Plotly charts. The local font family uses Noto Sans KR
DemiLight for body text, with Medium/Bold files for the existing weight hierarchy.

Typography should be compact and professional.

Avoid oversized SaaS-style typography.

Recommended scale:

Element	Size
Page title	24-28px
Section title	18-20px
Card title	14-16px
KPI value	26-32px
KPI label	12-14px
Normal body	14px
Secondary text	12-13px
Table text	13-14px

Avoid:

40px+
50px+
60px+

headings inside the dashboard.

This is an operational application, not a marketing website.

Use medium or semibold weights to create hierarchy
instead of excessively increasing font size.

# 11. Spacing System

Use a consistent spacing system.

Recommended values:

4px
8px
12px
16px
20px
24px
32px

Primary page padding:

24px

Card-to-card spacing:

16px - 20px

Internal card padding:

16px - 20px

Do not create excessive 40-80px whitespace
between normal dashboard components.

# 12. Border Radius

Use restrained enterprise-style rounding.

Allowed radius system:

8px
12px
16px

Recommended:

Inputs       8px
Buttons      8px
Small cards  8px
Main cards   12px
Modal/Drawer 12-16px

Avoid excessive:

24px
32px
999px

radii except for small badges where pill shapes are appropriate.

# 13. Shadows

Shadows should be subtle.

Cards should primarily be separated using:

spacing
surface contrast
light borders

rather than heavy shadows.

Example:

box-shadow:
    0 1px 2px rgba(16, 24, 40, 0.04),
    0 2px 8px rgba(16, 24, 40, 0.04);

Avoid:

strong floating-card shadows
neon shadows
colored shadows
excessive elevation
# 14. Dashboard Information Hierarchy

The visual hierarchy should be:

Level 1

Operational status and AI recommendation

Example:

D+3 발주량 예측
Level 2

Core KPI summary

Example:

AI 예상 발주량
기존 계획량
계획 대비 증감
검토 필요 Part
모델 MAE
Level 3

Primary forecast visualization

The main forecast chart must receive the strongest visual emphasis.

For 예지보전 and 품질 보증, the primary evidence chart replaces the forecast
chart at this hierarchy level.

Level 4

Actionable exceptions

Example:

우선 검토 부품
Level 5

Detailed supporting information

Example:

모델 정보
데이터 품질
연관 부품
입력 변수

## 14.1 Required Integrated Overview Tab Content

The integrated overview must support both operational demonstration and project
evaluation without becoming a slide deck.

### 통합 요약

Required content:

- Three track status cards
- Latest prediction and anomaly counts
- Data refresh time
- Cross-track priority action queue
- Recent action trend
- One concise key conclusion per track

### 프로젝트 개요

Required content:

- One-sentence project definition
- Three business questions and success criteria
- Three KAMP datasets and their distinct analysis units
- Explanation that the datasets are not merged row-by-row
- Mapping of time-series, supervised, and unsupervised learning to the three tracks
- Common workflow from data quality inspection to Dash usage
- KAMP source attribution

Use compact diagrams, tables, and summary cards. Avoid long report paragraphs.

### 데이터 품질

Required content:

- Track selector
- File-level row and column counts
- Missing values and duplicate rows
- Time interval and ordering checks
- Valid range and constant-column checks where applicable
- Before-and-after processing indicators
- Analysis unit and group counts
- Data dictionary access
- Download of the filtered quality result

This is a shared cross-track view. Do not duplicate a full data-quality page in
every track. A track page may still open a contextual quality Modal or Drawer.

### 결과 조회

Required content:

- Track selector
- Raw, processed, prediction, and evaluation result type selector
- Filterable data table
- Model version
- Data version
- Execution timestamp
- Evaluation-set identity
- Filtered result download
- Compact reproducibility and error-recovery guidance

### 결론·한계

Required content:

- Key business conclusion for each track
- Operational value
- Model-selection rationale
- Improvement or tradeoff versus baseline
- Data and model limitations
- Field-use cautions
- Future improvements
- Team contribution summary
- Final deliverable summary

This tab is the closing screen of the live demonstration and replaces the summary
normally delivered in a separate presentation.

## 14.2 Required Track Model Validation Tabs

Every track must have a dedicated `모델 검증` tab. A small model-information
Drawer is not sufficient evidence for project evaluation.

### 공급망 예측 모델 검증

Required content:

- Time-ordered train, validation, and test split
- Leakage-prevention rule
- Naive or moving-average baseline
- Candidate model comparison
- MAE and RMSE
- WAPE or SMAPE where valid
- Fold-level or walk-forward performance
- Part-level error analysis
- Over-forecast and under-forecast analysis
- Demand-spike behavior
- Final model-selection rationale
- Short observation-period limitation

### 예지보전 모델 검증

Required content:

- File or welding-sequence group split
- Leakage-prevention rule
- Locked independent test set
- Supervised candidate comparison
- Unsupervised candidate comparison
- Same-test-set comparison between the two model families
- Precision, Recall, and F1
- Confusion matrix and classification report
- Feature importance or explanation
- Anomaly score and threshold source
- Detection delay
- False-positive and false-negative interpretation
- Final operating-model rationale and limitation

### 품질 보증 모델 검증

Required content:

- Test, file, or battery-pack group split
- Evidence that rows from one test do not cross dataset splits
- Baseline classifier and candidate comparison
- Precision, Recall, and F1
- Defect Recall as a primary metric
- Confusion matrix and classification report
- Class-imbalance handling
- Defect-type performance
- Important voltage and temperature features
- Feature importance or explanation
- Final model-selection rationale
- Limitation of combining multiple defect types into one NG class

The operational 품질 분석 view may use PCA-based evidence, but the model validation
tab must separately present the supervised quality-classification evidence required
by the project evaluation criteria.
# 15. KPI Cards

KPI cards must remain compact.

Recommended desktop composition:

4-6 cards per row

depending on available content.

KPI cards should NOT resemble large mobile widgets.

Each KPI card should contain only:

Label
Main value
Optional comparison
Optional compact status

Do not place large illustrations inside KPI cards.

Do not make every KPI card a different color.

Prefer:

White card
+
neutral typography
+
small LS color accent
# 16. Filters

Filters should be grouped into a horizontal desktop toolbar.

Example:

[기준일 ▼] [Part ▼] [모델 ▼] [예측 기간 ▼]       [데이터 정상]

Do NOT create:

기준일
[     ]

Part
[     ]

모델
[     ]

as a long vertical mobile form unless absolutely necessary.

Filters should stay close to the visualization they control.

Shared context controls must remain outside the individual tab panels when they
affect more than one tab. Tab-local search, review filters, decision controls, and
export actions belong inside the relevant tab.

Switching secondary tabs must not reset shared filters, the selected Part, selected
event, selected cell or module, or the current review and decision state.

# 17. Buttons

Buttons should remain compact.

Recommended heights:

32px
36px
40px maximum for standard dashboard actions

Primary buttons:

LS Blue
#0A1E5A

Dangerous / critical actions:

LS Red
#FA002D

Do not use LS Red for normal primary actions.

Avoid large 48-56px mobile-style buttons.

Avoid full-width buttons unless the context specifically requires them.

# 18. Cards

Cards should group related operational information.

Use:

White background
Minimal border
Small shadow if necessary
8-12px radius
16-20px padding

Avoid unnecessary nested cards.

Bad:

Card
 └ Card
    └ Card
       └ Chart

Good:

Forecast Card
 ├ Header
 ├ Controls
 └ Chart
# 19. Plotly Chart Rules

Charts are analytical tools, not decoration.

Required
Remove unnecessary modebar buttons
Use unified hover where useful
Use subtle grid lines
Reduce chart ink
Use readable legends
Keep axes clean
Keep chart titles concise
Ensure tooltips provide actual values
Use consistent colors between screens

Recommended:

hovermode="x unified"
# 20. Forecast Chart Color Semantics

The same concept MUST use the same color throughout the application.

Recommended mapping:

AI Forecast
LS Blue
#0A1E5A

Actual Order
Dark neutral
#172033

Existing D+3 Plan
LS Light Blue or Gray
#0569A0 / #7D8282

Critical deviation
LS Red
#FA002D

Do not randomly assign chart colors.

Do not use rainbow chart palettes.

# 21. Main Forecast Visualization

The forecast chart is the primary visualization.

It should generally occupy more horizontal space than
secondary model information.

Example desktop structure:

┌─────────────────────────────────────────────────────────┐
│ Filters                                                 │
├──────────┬──────────┬──────────┬──────────┬──────────────┤
│ KPI      │ KPI      │ KPI      │ KPI      │ KPI          │
├──────────────────────────────────────┬──────────────────┤
│                                      │ Model / Insight  │
│        Forecast Chart                │ Panel            │
│                                      │                  │
├──────────────────────────────────────┴──────────────────┤
│ Review Required Parts                                  │
│ Dash AG Grid                                            │
└─────────────────────────────────────────────────────────┘

The chart should visually dominate the page.

# 22. Dash AG Grid

Operational tables should use Dash AG Grid.

Tables must support where appropriate:

Sorting
Filtering
Column resize
Row selection
Fixed header
Conditional formatting

Avoid overly tall rows.

Recommended row height:

36-42px

Important exceptions may use:

Light red background
Red indicator
Warning icon

but not full saturated red rows.

# 23. Actionable Exception Design

Only actionable exceptions should receive strong visual emphasis.

Example:

Prediction:
1,240

Plan:
820

Difference:
+420 (+51.2%)

When this requires review:

Soft warning background
+
small LS Red indicator
+
"검토 필요" badge

Do NOT paint the entire screen red.

# 24. Drawer

Detailed Part analysis should open inside a Drawer.

Do not navigate to another page unless necessary.

Recommended width on a 1920px desktop:

480px - 600px

The Drawer may contain:

Part Number
Current D+3 plan
AI forecast
Difference
Recommended quantity
Historical trend
Input variables
Related parts
Model explanation
# 25. Iconography

Use:

dash-iconify

Icons should improve scanning.

Recommended sizes:

16px
18px
20px

Avoid oversized decorative icons.

Do not put an icon inside every possible label.

# 26. Status Badges

Badges should be compact.

Examples:

정상
검토 필요
주의
위험
예측 완료
데이터 최신

Use low-saturation backgrounds with strong foreground colors.

Example:

Critical

background: soft red
text: LS Red

instead of a fully saturated red rectangle.

# 27. Empty States

Empty states should remain compact.

Do NOT create giant illustrations.

Use:

Small icon
Short description
Optional action

Example:

선택한 조건에서 검토가 필요한 Part가 없습니다.
# 28. Loading States

AI inference or data loading should clearly indicate status.

Prefer:

Skeleton
subtle loader
progress indicator

Avoid excessive animated effects.

The application should feel industrial and reliable,
not playful.

# 29. Animation

Animation must be restrained.

Allowed:

150-250ms transition
hover
drawer open
modal open
tooltip
small status transition

Avoid:

bouncing
glowing
rotating decorations
large entrance animations
parallax
excessive number animations
# 30. Responsive Behavior

Responsive behavior exists to preserve usability on smaller desktop monitors.

It is NOT permission to redesign the application as mobile.

On smaller desktop widths:

Reduce horizontal gap slightly
Allow KPI cards to wrap intelligently
Reduce secondary panel width if necessary
Preserve the main forecast chart width
Keep tables horizontally usable

Sidebar behavior:

- 1920px and larger: 240px sidebar
- 1600px - 1919px: 224px - 240px sidebar
- 1366px - 1599px: 216px sidebar
- Keep the text labels visible at all supported desktop resolutions
- Do not replace the sidebar with a mobile hamburger navigation in the primary scope
- Allow the main page padding to reduce from 24px to 20px or 16px as needed

Secondary tabs may scroll horizontally only as a last resort below the supported
desktop widths. They must fit without scrolling at 1920px, 1600px, and 1440px.

Do NOT immediately convert everything into a single vertical column.

The primary visual target remains desktop.

# 31. CSS Architecture

Reusable design variables must be located in:

assets/theme.css

Avoid large inline style dictionaries such as:

style={
    "background": "...",
    "padding": "...",
    "margin": "...",
    "border": "...",
    ...
}

repeated throughout Python components.

Prefer reusable classes.

Example:

className="kpi-card"

and:

.kpi-card {
    ...
}
# 32. Component Architecture

Create reusable Dash components.

Recommended structure:

components/
├── app_shell.py
├── sidebar.py
├── secondary_tabs.py
├── utility_header.py
├── filter_bar.py
├── kpi_card.py
├── forecast_chart.py
├── review_grid.py
├── part_drawer.py
├── status_badge.py
├── model_validation.py
├── data_quality_view.py
├── result_browser.py
├── project_overview.py
├── conclusion_summary.py
└── ai_copilot_entry.py

Do not duplicate visual implementation across pages.

The four primary navigation items, active-state behavior, secondary-tab styling,
and AI Copilot entry must be defined through shared components or shared
configuration. Do not maintain three visually divergent copies of the navigation.

# 33. dash-mantine-components Usage

Prefer Dash Mantine Components for common UI.

Examples:

dmc.AppShell
dmc.Container
dmc.Paper
dmc.Card
dmc.Group
dmc.Stack
dmc.SimpleGrid
dmc.Grid
dmc.Button
dmc.ActionIcon
dmc.Badge
dmc.Select
dmc.DatePickerInput
dmc.Tabs
dmc.Drawer
dmc.Modal
dmc.Tooltip
dmc.Skeleton
dmc.Alert

Do not recreate common controls manually with raw HTML
unless there is a clear technical reason.

# 34. Visual QA Requirements

After any major UI redesign:

Run the Dash application.
Open the application in a browser.
Capture the UI at 1920x1080.
Inspect the actual screenshot.
Check spacing.
Check text scale.
Check chart proportions.
Check card density.
Check table readability.
Check the sidebar width and fixed positioning.
Check that exactly one primary navigation item is active.
Check the active secondary tab on every domain.
Check that all required tab labels fit on one line.
Check shared filter state after switching tabs.
Check selected Part, event, cell, and decision state after switching tabs.
Check that AI Copilot remains near the bottom of the sidebar.
Check that all four primary navigation items open a working destination.
Check the integrated overview at all five required secondary tabs.
Check all three model-validation tabs against the project evaluation criteria.
Check that the screen does NOT resemble mobile/tablet UI.
Fix visual issues.
Repeat until visually consistent.

Do not consider UI implementation complete
only because the Python code runs successfully.

## 34.1 Live Demonstration Flow

The recommended live demonstration sequence is:

1. 프로젝트 개요
2. 통합 요약
3. 공급망 예측: 예측 분석 → 검토 목록 → 모델 검증
4. 예지보전: 이상 분석 → 이벤트 검토 → 모델 검증
5. 품질 보증: 품질 분석 → 판정·조치 → 모델 검증
6. 데이터 품질
7. 결과 조회
8. 결론·한계

The design must make this flow easy to follow without requiring a separate slide deck.
Each evaluation view should lead with the conclusion and then expose supporting
evidence. Avoid forcing the presenter to open local CSV, image, or report files during
the primary demonstration.

# 35. Final Design Checklist

Before completing UI work verify:

 Plotly Dash remains the frontend framework
 Layout is designed for 1920px desktop
 Fixed left sidebar is 240px at 1920px
 Main workspace uses the remaining 1680px
 Main page padding is 24px at 1920px
 Sidebar remains visible at supported desktop sizes
 Sidebar contains 통합 현황, 공급망 예측, 예지보전, 품질 보증 in that order
 AI Copilot is separated from primary navigation and anchored near the bottom
 Top global navigation is not duplicated above the main workspace
 Required secondary tabs match the approved information architecture
 Shared filters persist when switching tabs
 Primary navigation opens working destinations without URL or port collision
 LS Blue is used as primary brand color
 LS Red is used sparingly
 Background is light and neutral
 Cards are compact
 Typography is not oversized
 Buttons are not oversized
 Main forecast chart has highest visual priority
 KPI cards are secondary
 Review table uses available width
 Actionable anomalies are easy to identify
 Plotly charts use consistent semantic colors
 Tables support sorting/filtering
 Part detail uses Drawer
 Event detail uses Drawer
 Model information remains a contextual Drawer or Modal
 통합 현황 includes 통합 요약, 프로젝트 개요, 데이터 품질, 결과 조회, 결론·한계
 공급망 예측 includes 예측 분석, 검토 목록, 모델 검증
 예지보전 includes 이상 분석, 이벤트 검토, 모델 검증
 품질 보증 includes 품질 분석, 판정·조치, 모델 검증
 공급망 model validation shows time split, baseline comparison, MAE, and RMSE
 예지보전 model validation shows same-test-set supervised and unsupervised comparison
 품질 보증 model validation shows supervised classification, group split, Recall, and F1
 Data quality view includes before-and-after evidence for all three tracks
 Result view includes model version, data version, execution time, and download
 결론·한계 includes conclusions, limitations, improvements, and team contributions
 UI does not resemble a mobile application
 UI does not resemble a tablet application
 No excessive gradients
 No rainbow charts
 No excessive shadows
 No unnecessary decorative cards
 Reusable CSS exists in assets/theme.css
 Reusable UI components are implemented
