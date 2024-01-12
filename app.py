import streamlit as st
from transformers import BertModel, BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import requests
import re
import os


# 從環境變量中獲取 API keys，並將其轉換為列表
api_keys = os.environ['API_KEYS'].split(',')

# 從環境變量中獲取 search engine ID
search_engine_id = os.environ['SEARCH_ENGINE_ID']

# 初始化BERT模型和分詞器
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertModel.from_pretrained('bert-base-chinese')

# 定義切割段落的函數
def split_paragraph(paragraph):
    # 首先根據換行符號切割段落
    paragraphs = paragraph.split('\n')
    split_paragraphs = []

    # 然後檢查每個段落的長度
    for para in paragraphs:
        if len(para) > 90:
            # 如果段落超過90個字符，進一步切割
            split_points = [m.start() for m in re.finditer(r'[，。？]', para)]
            while split_points:
                # 找到最接近90個字符的切割點
                split_points_less_than_90 = [p for p in split_points if p < 90]
                if split_points_less_than_90:
                    split_point = max(split_points_less_than_90)
                    # 切割段落並加入到結果列表
                    split_paragraphs.append(para[:split_point+1])
                    para = para[split_point+1:]
                    # 更新切割點
                    split_points = [m.start() for m in re.finditer(r'[，。？]', para)]
        if para:
            # 加入最後一個段落或原本就不需要切割的段落
            split_paragraphs.append(para)

    return split_paragraphs

# 定義移除日期和省略號的函數
def clean_snippet(snippet):
    # 移除日期，匹配如 "Jun 13, 2023" 的格式
    snippet = re.sub(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s\d{1,2},\s\d{4}\b', '', snippet)
    # 移除省略號
    snippet = snippet.replace('...', '').strip()
    return snippet


# 定義進行查詢的函數
def search_query(query, api_keys, search_engine_id):
    # 清理查詢
    query = re.sub(r"^\d+\.", "", query)  # 移除開頭的數字列點
    query = re.sub(r"^Q\d*：", "", query)  # 移除開頭的 "Q：" 或 "Q1：" 等
    query = re.sub(r"^A：", "", query)  # 移除開頭的 "A："
    query = query.strip()  # 移除前後的空白字符

    for key in api_keys:
        try:
            response = requests.get(
                'https://customsearch.googleapis.com/customsearch/v1',
                params={
                    'q': query,
                    'key': key,
                    'cx': search_engine_id,  # 加入搜尋引擎ID
                    'lr': 'lang_zh-TW',      # 指定語言為繁體中文
                    'cr': 'countryTW',       # 指定地區為台灣
                    'gl': 'TW'               # 指定地理位置為台灣
                }
            )
            if response.status_code == 200:
                results = response.json().get('items')
                if results:
                    title = results[0]['title']
                    url = results[0]['link']
                    snippet = clean_snippet(results[0]['snippet'])
                    return title, url, snippet
        except requests.exceptions.RequestException as e:
            continue  # 如果當前API key額度用盡，則繼續嘗試下一個key
    return None, None, None

def calculate_similarity(text1, text2):
    # 將文本轉換為模型的輸入
    inputs1 = tokenizer(text1, return_tensors="pt", padding=True, truncation=True)
    inputs2 = tokenizer(text2, return_tensors="pt", padding=True, truncation=True)

    # 獲取文本的嵌入表示
    with torch.no_grad():
        outputs1 = model(**inputs1)
        outputs2 = model(**inputs2)

    # 取出文本的嵌入向量
    sentence_embedding1 = outputs1.last_hidden_state.mean(dim=1).numpy()
    sentence_embedding2 = outputs2.last_hidden_state.mean(dim=1).numpy()

    # 計算餘弦相似度
    similarity = cosine_similarity(sentence_embedding1, sentence_embedding2)

    # cosine_similarity 返回的是一個矩陣，我們需要的是兩個向量之間的相似度
    return similarity[0][0] * 100

# Streamlit界面
st.header("文案抄襲檢查工具", divider='rainbow')


# 輸入文案的大型文字輸入框
st.markdown('''
    :rainbow[請輸入想檢查的文案]''')
input_text = st.text_area('↓↓↓↓↓↓↓↓↓↓', height=300)

# 檢查文案是否抄襲按鈕
if st.button('檢查文案是否抄襲'):
    with st.spinner('正在檢查中，請稍候...'):
        # 切割段落
        paragraphs = split_paragraph(input_text)
        plagiarism_detected = False  # 用於標記是否檢測到抄襲
        checked_paragraphs_count = 0  # 計數器，用於計算檢查了多少個段落

        # 遍歷文檔中的每個段落
        for paragraph in paragraphs:
            if len(paragraph) > 50:
                checked_paragraphs_count += 1  # 更新檢查段落的計數器
                # 截取前70個字符作為查詢條件
                query = paragraph[:70] if len(paragraph) > 70 else paragraph

                # 進行查詢，確保傳遞search_engine_id
                title, url, snippet = search_query(query, api_keys, search_engine_id)
                if snippet:
                    # 移除snippet開頭的日期
                    snippet = clean_snippet(snippet)

                    # 計算相似度，這裡比較的是查詢條件和snippet
                    similarity = calculate_similarity(query, snippet)
                    
                    # 僅保留相似度大於95%的內容
                    if similarity > 95:
                        plagiarism_detected = True  # 標記為檢測到抄襲
                        # 呈現結果
                        st.write('檢查段落：', paragraph)  # 顯示當前正在檢查的段落
                        st.write('Google 返回的 descrption：', snippet)  # 顯示查詢到的相似文字
                        st.markdown(f'[{title}]({url})')  # 顯示標題錨文字和URL
                        
                        # 顯示相似度
                        if similarity > 97:
                            st.markdown(f'內容相似度：{similarity}%，<span style="color:red; font-weight:bold;">疑似抄襲，請檢查！！！！！</span>', unsafe_allow_html=True)
                        else:
                            st.write(f'內容相似度：{similarity}%')
                        
                        st.write('========================')  # 顯示分隔號

        # 如果所有段落檢查完畢且未檢測到抄襲
        if not plagiarism_detected:
            st.header('本次文字未檢查出抄襲文字')

        # 顯示檢查段落的總數
        st.header(f'本次一共檢查{checked_paragraphs_count}個段落')
        for i, paragraph in enumerate(paragraphs, 1):
            if len(paragraph) > 50:
                st.write(f'{i}. {paragraph}')
    st.success('檢查完成！')
