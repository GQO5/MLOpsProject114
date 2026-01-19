# NutriScan AI Color Palette (Extracted from actual code)
COLORS = {
    # Primary colors (exact from tailwind config)
    "primary": "#305d5f",
    "primary_dark": "#234647",
    "accent_coral": "#F7A072",
    # Backgrounds
    "background_light": "#f9fbfb",
    "card_light": "#ffffff",
    # Text
    "text_main": "#131616",
    "text_muted": "#6d7d7e",
    # Nutrient card icon backgrounds (exact from HTML)
    "calories_bg": "rgba(247, 160, 114, 0.2)",  # accent-coral/20
    "calories_icon": "#F7A072",  # accent-coral
    "protein_bg": "rgba(48, 93, 95, 0.2)",  # primary/20
    "protein_icon": "#305d5f",  # primary
    "carbs_bg": "rgb(254, 249, 195)",  # yellow-100
    "carbs_icon": "rgb(202, 138, 4)",  # yellow-600
    "fat_bg": "rgb(219, 234, 254)",  # blue-100
    "fat_icon": "rgb(37, 99, 235)",  # blue-600
    # Border
    "border": "rgb(241, 243, 243)",
    "border_gray": "rgb(243, 244, 246)",
}

# CSS template for nutrition cards (exact styling from HTML)
NUTRIENT_CARD_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@200..800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap');
    * {
        font-family: 'Manrope', sans-serif;
    }
    .nutrient-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
        margin-top: 24px;
    }
    @media (max-width: 640px) {
        .nutrient-container {
            grid-template-columns: repeat(2, 1fr);
        }
    }
    .nutrient-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 20px -2px rgba(48, 93, 95, 0.08);
        border: 1px solid rgb(243, 244, 246);
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        justify-content: space-between;
        position: relative;
        overflow: hidden;
        transition: transform 0.3s ease;
    }
    .nutrient-card:hover {
        transform: translateY(-2px);
    }
    .nutrient-card-bg {
        position: absolute;
        right: -16px;
        top: -16px;
        width: 80px;
        height: 80px;
        border-radius: 50%;
        transition: transform 0.3s ease;
    }
    .nutrient-card:hover .nutrient-card-bg {
        transform: scale(1.25);
    }
    .nutrient-icon {
        padding: 8px;
        border-radius: 8px;
        margin-bottom: 12px;
        z-index: 10;
        position: relative;
    }
    .material-symbols-outlined {
        font-variation-settings: 'FILL' 1, 'wght' 400, 'GRAD' 0, 'opsz' 24;
    }
    .nutrient-label {
        color: #6d7d7e;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        z-index: 10;
        position: relative;
    }
    .nutrient-value {
        color: #131616;
        font-size: 32px;
        font-weight: 800;
        line-height: 1;
        margin-top: 4px;
        z-index: 10;
        position: relative;
    }
    .nutrient-unit {
        font-size: 18px;
        font-weight: 700;
        color: #6d7d7e;
        vertical-align: top;
    }
    .nutrient-sublabel {
        font-size: 12px;
        color: #6d7d7e;
        margin-top: 2px;
        z-index: 10;
        position: relative;
    }
</style>
"""


def get_header_html():
    """
    Generate HTML for header matching NutriScan AI design
    """
    return """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@200..800&display=swap');
            @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap');
            /* Header styling */
            .custom-header {
                position: sticky;
                top: 0;
                z-index: 999;
                background: rgba(255, 255, 255, 0.9);
                backdrop-filter: blur(10px);
                border-bottom: 1px solid #f1f3f3;
                padding: 16px 0;
                margin-bottom: 2px;
            }
            .header-content {
                max-width: 1200px;
                margin: 0 auto;
                padding: 0 24px;
                display: flex;
                align-items: center;
                justify-content: space-between;
            }
            .logo-section {
                display: flex;
                align-items: center;
                gap: 12px;
            }
            .logo-icon {
                width: 32px;
                height: 32px;
                border-radius: 8px;
                background: rgba(48, 93, 95, 0.1);
                display: flex;
                align-items: center;
                justify-content: center;
                color: #305d5f;
            }
            .logo-text {
                font-size: 20px;
                font-weight: 700;
                color: #305d5f;
                font-family: 'Manrope', sans-serif;
                letter-spacing: -0.02em;
            }
            .header-actions {
                display: flex;
                align-items: center;
                gap: 12px;
            }
            .settings-btn {
                width: 36px;
                height: 36px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                background: transparent;
                border: none;
                cursor: pointer;
                color: #6d7d7e;
                transition: background 0.2s;
            }
            .settings-btn:hover {
                background: rgba(0, 0, 0, 0.05);
            }
            .user-avatar {
                width: 32px;
                height: 32px;
                border-radius: 50%;
                background: linear-gradient(135deg, #305d5f, #F7A072);
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-weight: 700;
                font-size: 12px;
                font-family: 'Manrope', sans-serif;
                box-shadow: 0 2px 8px rgba(48, 93, 95, 0.2);
            }
            .material-symbols-outlined {
                font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
            }
            /* Center content with max width */
            .main-content {
                max-width: 1200px;
                margin: 0 auto;
                padding: 0 24px;
            }
            /* Remove default Streamlit padding */
            .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
                max-width: 1200px;
            }
            /* Center image vertically */
            [data-testid="column"]:first-child {
                display: flex;
                align-items: center;
                justify-content: center;
            }
            [data-testid="stImage"] {
                display: flex;
                align-items: center;
                justify-content: center;
            }
        </style>

        <div class="custom-header">
            <div class="header-content">
                <div class="logo-section">
                    <div class="logo-icon">
                        <span class="material-symbols-outlined">nutrition</span>
                    </div>
                    <h1 class="logo-text">NutriScan AI</h1>
                </div>
                <div class="header-actions">
                    <button class="settings-btn">
                        <span class="material-symbols-outlined">settings</span>
                    </button>
                    <div class="user-avatar">JD</div>
                </div>
            </div>
        </div>
        """


def get_nutrient_card_html(total_calories, total_fat, total_carbs, total_protein):
    """
    Generate HTML for nutrition cards matching NutriScan AI design exactly
    """
    return f"""
    {NUTRIENT_CARD_CSS}
    <div class="nutrient-container">
        <!-- Calories Card -->
        <div class="nutrient-card">
            <div class="nutrient-card-bg" style="background-color: rgba(247, 160, 114, 0.1);"></div>
            <div class="nutrient-icon" style="background-color: rgba(247, 160, 114, 0.2); color: #F7A072;">
                <span class="material-symbols-outlined">local_fire_department</span>
            </div>
            <div>
                <p class="nutrient-label">Calories</p>
                <p class="nutrient-value">{int(total_calories)}<span class="nutrient-unit">kcal</span></p>
            </div>
        </div>
        <!-- Protein Card -->
        <div class="nutrient-card">
            <div class="nutrient-card-bg" style="background-color: rgba(48, 93, 95, 0.1);"></div>
            <div class="nutrient-icon" style="background-color: rgba(48, 93, 95, 0.2); color: #305d5f;">
                <span class="material-symbols-outlined">fitness_center</span>
            </div>
            <div>
                <p class="nutrient-label">Protein</p>
                <p class="nutrient-value">{total_protein:.2f}<span class="nutrient-unit">g</span></p>
            </div>
        </div>
        <!-- Carbs Card -->
        <div class="nutrient-card">
            <div class="nutrient-card-bg" style="background-color: rgba(234, 179, 8, 0.1);"></div>
            <div class="nutrient-icon" style="background-color: rgb(254, 249, 195); color: rgb(202, 138, 4);">
                <span class="material-symbols-outlined">grain</span>
            </div>
            <div>
                <p class="nutrient-label">Carbs</p>
                <p class="nutrient-value">{total_carbs:.2f}<span class="nutrient-unit">g</span></p>
            </div>
        </div>
        <!-- Fat Card -->
        <div class="nutrient-card">
            <div class="nutrient-card-bg" style="background-color: rgba(59, 130, 246, 0.1);"></div>
            <div class="nutrient-icon" style="background-color: rgb(219, 234, 254); color: rgb(37, 99, 235);">
                <span class="material-symbols-outlined">water_drop</span>
            </div>
            <div>
                <p class="nutrient-label">Fat</p>
                <p class="nutrient-value">{total_fat:.2f}<span class="nutrient-unit">g</span></p>
            </div>
        </div>
    </div>
    """
