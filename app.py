"""Streamlit-сервис прогнозирования Engagement Rate для блогеров."""
import streamlit as st
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns, os, json
from catboost import CatBoostRegressor
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="ER Predictor", layout="wide", initial_sidebar_state="collapsed")

# Стиль
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

MODEL_PATH = "data/processed/catboost_err_model.cbm"
DATA_PATH = "data/raw/all_posts_raw.csv"
META_PATH = "data/processed/model_meta.json"

@st.cache_data
def load_data():
    raw = pd.read_csv(DATA_PATH)
    raw = raw[raw["views_count"] > 0]
    raw["ERR"] = np.where(raw["views_count"] > 0,
        (raw["likes_count"] + raw["comments_count"] + raw["forwards_count"]) / raw["views_count"] * 100, np.nan)
    raw = raw[raw["ERR"] <= 100]
    raw["published_dt"] = pd.to_datetime(raw["published_at"], utc=True)
    ch = raw.groupby("channel_username").agg(
        followers=("followers_count", "first"),
        platform=("platform", "first"),
        channel_title=("channel_title", "first"),
        posts=("post_id", "count"),
        avg_views=("views_count", "mean"),
        avg_likes=("likes_count", "mean"),
        avg_forwards=("forwards_count", "mean"),
        avg_comments=("comments_count", "mean"),
        avg_text_len=("text_length", "mean"),
        ad_count=("is_advert", "sum"),
        avg_err=("ERR", "mean"),
        median_err=("ERR", "median"),
    ).reset_index()
    return raw, ch

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    m = CatBoostRegressor()
    m.load_model(MODEL_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    return m, meta

# ==================== Helper Functions ====================

def get_market_metrics(raw: pd.DataFrame):
    """Рыночные метрики."""
    return {
        'avg_err': raw['ERR'].mean(),
        'median_err': raw['ERR'].median(),
        'avg_views': raw['views_count'].mean(),
        'avg_text_len': raw['text_length'].mean(),
        'ad_ratio': (raw['is_advert'].sum() / len(raw) * 100) if len(raw) > 0 else 0,
    }

def generate_insights(channel_name: str, ch_data: dict, ch_raw: pd.DataFrame, raw: pd.DataFrame) -> list:
    """Генерирует аналитические инсайты по каналу."""
    insights = []
    market = get_market_metrics(raw)
    
    # ERR vs рынок
    err_diff = ch_data['avg_err'] - market['avg_err']
    if abs(err_diff) > 0.5:
        sign = "выше" if err_diff > 0 else "ниже"
        insights.append(f"ERR канала {sign} среднего по рынку на {abs(err_diff):.1f}%")
    
    # Тренд за 30 дней
    if len(ch_raw) > 1:
        ch_raw_copy = ch_raw.copy()
        ch_raw_copy['date'] = ch_raw_copy['published_dt'].dt.date
        cutoff = ch_raw_copy['date'].max() - timedelta(days=30)
        daily = ch_raw_copy[ch_raw_copy['date'] >= cutoff].groupby('date')['ERR'].mean()
        if len(daily) >= 2:
            trend = daily.iloc[-1] - daily.iloc[0]
            trend_sign = "рост" if trend > 0 else "падение"
            insights.append(f"За последние 30 дней вовлечённость {trend_sign} на {abs(trend):.2f}%")
    
    # Реклама vs обычные посты
    ch_ad = ch_raw[ch_raw['is_advert'] == True]['ERR'].mean()
    ch_normal = ch_raw[ch_raw['is_advert'] == False]['ERR'].mean()
    if not np.isnan(ch_ad) and not np.isnan(ch_normal):
        ad_diff = ch_normal - ch_ad
        if ad_diff > 0.5:
            insights.append(f"Обычные посты показывают ERR на {ad_diff:.1f}% выше рекламных")
    
    # Платформа
    if ch_data['platform'] == 'Telegram':
        insights.append(f"Канал Telegram – требует более частых публикаций для стабильности")
    else:
        insights.append(f"Группа VK – аудитория зависит от алгоритма ленты")
    
    return insights if insights else ["Канал работает стабильно"]

def plot_heatmap_hours(raw: pd.DataFrame) -> plt.Figure:
    """Тепловая карта: день недели × час публикации."""
    raw_copy = raw.copy()
    raw_copy['hour'] = raw_copy['published_dt'].dt.hour
    raw_copy['dow'] = raw_copy['published_dt'].dt.day_name()
    raw_copy['dow_num'] = raw_copy['published_dt'].dt.dayofweek
    
    heatmap_data = raw_copy.pivot_table(
        values='ERR', index='hour', columns='dow_num', aggfunc='mean'
    )
    
    dow_labels = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']
    heatmap_data.columns = [dow_labels[int(c)] for c in heatmap_data.columns]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn', cbar_kws={'label': 'Средний ERR (%)'}, ax=ax, linewidths=0.5)
    ax.set_title('Тепловая карта: лучшее время для публикации', fontsize=12, fontweight='bold')
    ax.set_xlabel('День недели', fontsize=10)
    ax.set_ylabel('Час дня', fontsize=10)
    return fig

def plot_feature_importance(model, meta: dict) -> plt.Figure:
    """График Feature Importance из модели CatBoost."""
    try:
        # Получаем feature importance прямо из модели
        fi_data = model.get_feature_importance()
        if fi_data is None or len(fi_data) == 0:
            return None
        
        # Преобразуем в словарь с названиями фичей
        feature_names_list = meta.get('features', [])
        fi = {feature_names_list[i]: fi_data[i] for i in range(len(fi_data))}
    except:
        return None
    
    if not fi:
        return None
    
    # Берём топ-10
    sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:10]
    features = [item[0] for item in sorted_fi]
    importances = [item[1] for item in sorted_fi]
    
    # Переводим названия фичей
    feature_names = {
        'channel_median_err': 'Медиана ERR канала',
        'channel_avg_err': 'Средний ERR канала',
        'text_density': 'Плотность текста',
        'month': 'Месяц',
        'hour': 'Час публикации',
        'log_followers': 'Лог подписчиков',
        'is_advert': 'Рекламный пост',
        'text_length': 'Длина текста',
        'has_photo': 'Фото',
        'has_video': 'Видео',
        'followers_count': 'Подписчики',
        'n_hashtags': 'Хештеги',
        'n_links': 'Ссылки',
        'n_emojis': 'Эмодзи',
        'has_document': 'Документы',
        'has_any_media': 'Медиа',
        'has_link': 'Ссылка',
        'has_hashtag': 'Хештег',
        'is_telegram': 'Telegram',
        'channel_posts_total': 'Всего постов в канале',
        'channel_posts_per_day': 'Постов в день',
        'channel_avg_views': 'Средние просмотры',
        'channel_avg_text_len': 'Средняя длина текста',
        'clean_text_length': 'Чистая длина текста',
        'n_mentions': 'Упоминания',
        'is_weekend': 'Выходной',
        'day_of_week': 'День недели',
    }
    
    features = [feature_names.get(f, f) for f in features]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(features))
    ax.barh(y_pos, importances, color='steelblue', alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel('Важность (%)', fontsize=10)
    ax.set_title('Что влияет на Engagement Rate', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    return fig

def compare_with_market(ch_data: dict, raw: pd.DataFrame) -> pd.DataFrame:
    """Сравнение канала с рынком."""
    market = get_market_metrics(raw)
    
    comparison = pd.DataFrame({
        'Метрика': ['Средний ERR', 'Просмотры на пост', 'Длина текста', 'Доля рекламы'],
        'Канал': [
            f"{ch_data['avg_err']:.2f}%",
            f"{ch_data['avg_views']:,.0f}",
            f"{ch_data['avg_text_len']:.0f}",
            f"{(ch_data['ad_count']/ch_data['posts']*100):.1f}%",
        ],
        'Рынок': [
            f"{market['avg_err']:.2f}%",
            f"{market['avg_views']:,.0f}",
            f"{market['avg_text_len']:.0f}",
            f"{market['ad_ratio']:.1f}%",
        ],
        'Дельта': [
            f"{ch_data['avg_err'] - market['avg_err']:+.2f}%",
            f"{ch_data['avg_views'] - market['avg_views']:+,.0f}",
            f"{ch_data['avg_text_len'] - market['avg_text_len']:+.0f}",
            f"{(ch_data['ad_count']/ch_data['posts']*100 - market['ad_ratio']):+.1f}%",
        ]
    })
    return comparison

def predict_er(channel_name, text_len, hashtags, links, emojis,
               photo, video, advert, hour, dow, month=6):
    """Прогноз ERR с использованием CatBoost модели."""
    m, meta = load_model()
    if m is None: 
        return None
    
    raw, ch_agg = load_data()
    ch = ch_agg[ch_agg["channel_username"] == channel_name]
    if len(ch) == 0: 
        return None
    
    f = ch["followers"].iloc[0]
    td = text_len / (hashtags + links) if (hashtags + links) > 0 else text_len
    
    row = pd.DataFrame([{
        "followers_count": f, "log_followers": np.log1p(f),
        "hour": hour, "day_of_week": dow, "is_weekend": int(dow in [5,6]), "month": month,
        "text_length": text_len, "clean_text_length": text_len,
        "n_hashtags": hashtags, "n_links": links, "n_mentions": 0, "n_emojis": emojis,
        "has_photo": int(photo), "has_video": int(video), "has_document": 0,
        "has_any_media": int(photo or video), "has_link": int(links > 0),
        "has_hashtag": int(hashtags > 0), "is_advert": int(advert),
        "is_telegram": int(ch["platform"].iloc[0] == "Telegram"),
        "text_density": td,
        "channel_avg_err": ch["avg_err"].iloc[0],
        "channel_median_err": ch["median_err"].iloc[0],
        "channel_posts_total": ch["posts"].iloc[0],
        "channel_posts_per_day": ch["posts"].iloc[0] / 30,
        "channel_avg_views": ch["avg_views"].iloc[0],
        "channel_avg_text_len": ch["avg_text_len"].iloc[0],
    }])
    
    features = meta.get("features", [c for c in row.columns])
    row = row[[c for c in features if c in row.columns]]
    
    try:
        pred = np.expm1(m.predict(row)[0])
        return max(0, pred)  # Убедимся, что результат не отрицательный
    except Exception as e:
        print(f"[ERR] Prediction failed: {e}")
        return None

# ======================== UI ========================
st.title("ER Predictor — Engagement Rate Analytics")
st.markdown("Анализ вовлечённости и прогнозирование для блогеров", help="Использует CatBoost для предсказания")

raw, ch_agg = load_data()

tab1, tab2, tab3 = st.tabs(["Главная", "Анализ канала", "Прогноз"])

# ============================== TAB 1: DASHBOARD ==============================
with tab1:
    # KPI Row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Постов", f"{len(raw):,}", "в базе")
    col2.metric("Каналов", f"{raw['channel_username'].nunique():,}", "уникальных")
    col3.metric("Средний ERR", f"{raw['ERR'].mean():.2f}%", f"{raw['ERR'].std():.2f}% σ")
    col4.metric("Медианный ERR", f"{raw['ERR'].median():.2f}%", "устойчивая величина")
    
    st.divider()
    
    # Row 1: Distribution + ERR by platform
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Распределение ERR")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(raw["ERR"], bins=60, color="#00cc88", edgecolor="white", alpha=0.8)
        ax.axvline(raw["ERR"].mean(), color="red", linestyle="--", lw=2, label=f"Средн. = {raw['ERR'].mean():.2f}%")
        ax.axvline(raw["ERR"].median(), color="blue", linestyle="-.", lw=2, label=f"Медиана = {raw['ERR'].median():.2f}%")
        ax.set_xlabel("ERR (%)", fontsize=10)
        ax.set_ylabel("Постов", fontsize=10)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(alpha=0.3, axis='y')
        st.pyplot(fig)
    
    with col2:
        st.subheader("ERR по платформам")
        fig, ax = plt.subplots(figsize=(7, 4))
        platforms = raw.groupby("platform")["ERR"].agg(["mean", "median", "count"])
        x = range(len(platforms))
        ax.bar(x, platforms["mean"], color=["#0084ff", "#ff6b6b"], alpha=0.8, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(platforms.index, fontsize=11)
        ax.set_ylabel("Средний ERR (%)", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        for i, v in enumerate(platforms["mean"]):
            ax.text(i, v + 0.02, f"{v:.2f}%", ha="center", fontsize=11, fontweight="bold")
        st.pyplot(fig)
    
    st.divider()
    
    # Row 2: Largest channels and their ER
    st.subheader("Крупнейшие каналы и их ER")
    top = ch_agg.nlargest(15, "followers").copy()
    top["name"] = top["channel_title"].fillna(top["channel_username"]).str[:25]
    top["followers_m"] = (top["followers"] / 1_000_000).round(1)

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#ff6b6b" if p == "VK" else "#0084ff" for p in top["platform"]]
    bars = ax.barh(range(len(top)), top["avg_err"], color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([f"{n} ({f}M)" for n, f in zip(top["name"], top["followers_m"])], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Средний ERR (%)", fontsize=10)
    ax.grid(axis="x", alpha=0.3)
   
    st.pyplot(fig)
    
    st.divider()
    
    # Feature Importance
    st.subheader("Факторы успеха (Feature Importance)")
    if load_model() is not None:
        model, meta = load_model()
        fig = plot_feature_importance(model, meta)
        if fig:
            st.pyplot(fig)
            st.caption("Модель учитывает историю канала (54%), время публикации и характеристики поста")
        else:
            st.info("Данные о важности фичей недоступны")
    else:
        st.warning("Модель не загружена")

# ============================== TAB 2: CHANNEL ANALYSIS ==============================
with tab2:
    channel = st.selectbox("Выберите канал", sorted(ch_agg["channel_username"].unique()))
    
    if channel:
        ch = ch_agg[ch_agg["channel_username"] == channel].iloc[0]
        ch_raw = raw[raw["channel_username"] == channel].sort_values("published_dt")
        
        # Header
        title_str = ch.get("channel_title", channel)
        title_str = title_str if pd.notna(title_str) and title_str else channel
        platform_emoji = "Telegram" if ch["platform"] == "Telegram" else "VK"
        st.header(f"{title_str} ({platform_emoji})")
        
        # KPI metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Подписчики", f"{ch['followers']:,.0f}")
        col2.metric("Постов", int(ch["posts"]))
        col3.metric("Средний ERR", f"{ch['avg_err']:.2f}%")
        col4.metric("Медиана ERR", f"{ch['median_err']:.2f}%")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Ср. просмотры", f"{ch['avg_views']:,.0f}")
        col2.metric("Ср. длина", f"{ch['avg_text_len']:.0f}")
        col3.metric("Реклама", f"{ch['ad_count']/ch['posts']*100:.1f}%")

        col1, col2, col3 = st.columns(3)
        col1.metric("Ср. лайки", f"{ch['avg_likes']:,.0f}")
        col2.metric("Ср. репосты", f"{ch['avg_forwards']:,.0f}")
        col3.metric("Ср. комментарии", f"{ch['avg_comments']:,.0f}")
        
        st.divider()
        
        # Market comparison
        st.subheader("Сравнение с рынком")
        comparison = compare_with_market(ch, raw)
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        
        st.divider()
        
        # 30-day trend
        st.subheader("Тренд ERR за 30 дней")
        if len(ch_raw) > 0 and "published_dt" in ch_raw.columns:
            ch_raw_copy = ch_raw.copy()
            ch_raw_copy["date"] = ch_raw_copy["published_dt"].dt.date
            cutoff = ch_raw_copy["date"].max() - timedelta(days=30)
            daily = ch_raw_copy[ch_raw_copy["date"] >= cutoff].groupby("date")["ERR"].mean().reset_index()
            daily = daily.sort_values("date")
            
            if len(daily) >= 3:
                daily["smooth"] = daily["ERR"].rolling(window=3, min_periods=1, center=True).mean()
                trend_up = daily["smooth"].iloc[-1] >= daily["smooth"].iloc[0]
                delta_color = "#2ecc71" if trend_up else "#e74c3c"
                delta = daily["smooth"].iloc[-1] - daily["smooth"].iloc[0]
                arrow = "рост" if trend_up else "падение"
                
                fig, ax = plt.subplots(figsize=(14, 5))
                ax.fill_between(daily["date"], daily["smooth"], alpha=0.2, color=delta_color)
                ax.plot(daily["date"], daily["smooth"], "-", color=delta_color, lw=3, label="Тренд (скользящее среднее)")
                ax.set_xlabel("Дата", fontsize=10)
                ax.set_ylabel("ERR (%)", fontsize=10)
                ax.set_title(f"Динамика: {delta:+.2f}% {arrow}", fontsize=12, fontweight='bold')
                ax.legend(loc='best', fontsize=9)
                ax.grid(alpha=0.3)
                fig.autofmt_xdate()
                st.pyplot(fig)
                st.caption(f"{daily['date'].iloc[0]} — {daily['date'].iloc[-1]} | Изменение: {delta:+.2f}%")
            else:
                st.info("Недостаточно данных для анализа тренда")
        
        st.divider()
             
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Распределение ERR")
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(ch_raw["ERR"], bins=min(30, len(ch_raw)), color="#3498db", edgecolor="white", alpha=0.8)
            ax.axvline(ch["avg_err"], color="red", linestyle="--", lw=2, label=f"Ср. = {ch['avg_err']:.2f}%")
            ax.axvline(ch["median_err"], color="green", linestyle="-.", lw=2, label=f"Медиана = {ch['median_err']:.2f}%")
            ax.set_xlabel("ERR (%)", fontsize=10)
            ax.set_ylabel("Постов", fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3, axis='y')
            st.pyplot(fig)
        
        with col2:
            st.subheader("ЕRR по часам публикации")
            if 'published_dt' in ch_raw.columns:
                ch_raw_copy = ch_raw.copy()
                ch_raw_copy['hour'] = ch_raw_copy['published_dt'].dt.hour
                hourly = ch_raw_copy.groupby('hour')['ERR'].agg(['mean', 'count']).reset_index()
                hourly = hourly[hourly['count'] >= 2]  # Только с 2+ постами
                
                if len(hourly) > 0:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    colors = ['#2ecc71' if v == hourly['mean'].max() else '#3498db' for v in hourly['mean']]
                    ax.bar(hourly['hour'], hourly['mean'], color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
                    ax.axhline(ch['avg_err'], color='red', linestyle='--', lw=2, label=f"Средний = {ch['avg_err']:.2f}%")
                    ax.set_xlabel("Час дня", fontsize=10)
                    ax.set_ylabel("Средний ERR (%)", fontsize=10)
                    ax.set_xticks(range(0, 24, 2))
                    ax.legend(fontsize=9)
                    ax.grid(alpha=0.3, axis='y')
                    st.pyplot(fig)
                else:
                    st.info("Недостаточно данных по часам")

# ============================== TAB 3: PREDICTION ==============================
with tab3:
    st.subheader("Прогноз Engagement Rate")
    
    if load_model() is None:
        st.error("Модель не найдена. Обучите модель в 03_modeling.ipynb")
    else:
        _, meta = load_model()
        
        # Model info
        col1, col2, col3 = st.columns(3)
        col1.metric("Модель", "CatBoostRegressor")
        col2.metric("R²", f"{meta['metrics'].get('R2', 'N/A')}")
        col3.metric("MAE", f"{meta['metrics'].get('MAE_ERR%', 'N/A')}%")
        
        st.divider()
        
        # Channel selection
        channel = st.selectbox("Выберите канал для прогноза", 
                              sorted(ch_agg["channel_username"].unique()),
                              key="pred_channel")
        
        if channel:
            ch = ch_agg[ch_agg["channel_username"] == channel].iloc[0]
            ch_raw = raw[raw["channel_username"] == channel]
            
            st.markdown(f"### {ch.get('channel_title', channel)}")
            
            # Input parameters
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Текст и контент")
                pred_text = st.slider("Длина текста (символов)", 10, 3000, int(ch["avg_text_len"]), step=50)
                pred_hashtags = st.slider("Хештеги", 0, 20, 3)
                pred_links = st.slider("Ссылки", 0, 10, 2)
                pred_emojis = st.slider("Эмодзи", 0, 20, 2)
            
            with col2:
                st.markdown("#### Медиа и время")
                pred_photo = st.checkbox("Фото", True)
                pred_video = st.checkbox("Видео", False)
                pred_advert = st.checkbox("Рекламный пост", True)
                pred_hour = st.slider("Час публикации", 0, 23, 18)
                pred_dow = st.selectbox("День недели",
                    ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"], index=2)
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                predict_button = st.button("Предсказать ER", type="primary", use_container_width=True)
            
            if predict_button:
                dow_idx = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"].index(pred_dow)
                er = predict_er(channel, pred_text, pred_hashtags, pred_links, pred_emojis,
                               pred_photo, pred_video, pred_advert, pred_hour, dow_idx)
                
                if er and er > 0:
                    st.divider()
                    
                    # Results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"""
                        ### Результат
                        
                        **Прогноз ERR: `{er:.2f}%`**
                        
                        ---
                        **Сравнение:**
                        - Канал (средний): {ch['avg_err']:.2f}%
                        - Рынок (средний): {raw['ERR'].mean():.2f}%
                        - Вашего поста vs рынок: **{er - raw['ERR'].mean():+.2f}%**
                        """)
                    
                    with col2:
                        # Gauge-like visualization
                        fig, ax = plt.subplots(figsize=(7, 5))
                        
                        # Минимум и максимум
                        min_err, max_err = raw['ERR'].min(), raw['ERR'].max()
                        market_avg = raw['ERR'].mean()
                        
                        # Bar
                        ax.barh(['Прогноз'], [max_err], color='lightgray', alpha=0.3, height=0.3)
                        ax.barh(['Прогноз'], [er], color='#3498db' if er >= market_avg else '#e74c3c', height=0.3, alpha=0.9)
                        
                        # Lines for reference
                        ax.axvline(market_avg, color='orange', linestyle='--', lw=2, label=f'Рынок ({market_avg:.2f}%)')
                        ax.axvline(ch['avg_err'], color='green', linestyle='-', lw=2, label=f'Канал ({ch['avg_err']:.2f}%)')
                        
                        ax.set_xlim(0, max_err)
                        ax.set_xlabel("ERR (%)", fontsize=11)
                        ax.legend(loc='lower right', fontsize=9)
                        ax.set_ylim(-0.5, 0.5)
                        ax.set_yticks([])
                        ax.grid(axis='x', alpha=0.3)
                        
                        st.pyplot(fig)
                        
                        # Status
                        if er >= market_avg:
                            st.success(f"Прогноз выше среднего рынка на {er - market_avg:.2f}%")
                        else:
                            st.info(f"Прогноз ниже среднего на {abs(er - market_avg):.2f}%")
                else:
                    st.error("Не удалось сделать прогноз. Проверьте данные канала.")

            # ---- Forecast by days (auto params) ----
            st.divider()
            st.subheader("Прогноз ER по дням (авто-параметры)")

            if st.button("Рассчитать на сегодня / +7 / +30 дней", use_container_width=True):
                ch_raw_f = raw[raw["channel_username"] == channel]
                if len(ch_raw_f) == 0:
                    st.error("Нет данных")
                else:
                    txt = int(ch_raw_f["text_length"].median())
                    h = int(ch_raw_f["n_hashtags"].median()) if "n_hashtags" in ch_raw_f.columns else 2
                    l = int(ch_raw_f["n_links"].median()) if "n_links" in ch_raw_f.columns else 1
                    e = int(ch_raw_f["n_emojis"].median()) if "n_emojis" in ch_raw_f.columns else 1
                    ph = ch_raw_f["has_photo"].mean() > 0.5
                    vid = ch_raw_f["has_video"].mean() > 0.3

                    st.caption(f"Параметры: {txt} симв., {h} хештегов, {l} ссылок, "
                              f"{e} эмодзи, фото={'да' if ph else 'нет'}")

                    today = pd.Timestamp.now().date()
                    dows = ["Пн","Вт","Ср","Чт","Пт","Сб","Вс"]
                    rows = []
                    for days in [0, 7, 30]:
                        d = today + timedelta(days=days)
                        dow = d.weekday()
                        er = predict_er(channel, txt, h, l, e, ph, vid, True, 18, dow, d.month)
                        label = "Сегодня" if days == 0 else f"+{days} дн."
                        rows.append({"Дата": label, "День": dows[dow],
                                    "Число": d.strftime("%d.%m.%Y"),
                                    "Прогноз ERR": f"{er:.2f}%" if er else "—"})
                    st.table(pd.DataFrame(rows))

