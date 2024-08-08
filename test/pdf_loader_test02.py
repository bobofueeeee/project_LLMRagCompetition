import pdfplumber
from untils.textsplitter import ChineseTextSplitter
spliter = ChineseTextSplitter(pdf=True)

def read_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        result = []

        total_pages = len(pdf.pages)
        first_page = pdf.pages[0]
        first_page_text = first_page.extract_text()
        result.append({"header": "首页", "footer":"首页", "content": first_page_text})
        print(result)

        for page_number in range(1,total_pages):
            # 尝试识别页眉和页脚（这里假设页眉和页脚在页面文本的开头和结尾）
            # 注意：这只是一个非常粗糙的示例
            pdf_dict = {}
            # 读取整页文本
            page = pdf.pages[page_number]
            text = page.extract_text()



            if text:
                lines = text.strip().split('\n')
                if lines:
                    # 假设第一行是页眉
                    header = lines[0]
                    # print("可能的页眉:")
                    # print(header)
                    pdf_dict['header'] = header

                    # 假设最后一行是页脚
                    footer = lines[-1]
                    # print("可能的页脚:")
                    # print(footer)
                    pdf_dict['footer'] = footer

                # 去除页眉页脚
                text = text[len(header):-len(footer)]
                # text_tmp = ''
                # for ele in lines[1:-1]:
                #     text_tmp = text_tmp + ele

                # 调用函数，传入PDF文件路
            # print(text_tmp)
            pdf_dict['content'] = spliter.split_text1(text)

            # 识别表格
            tables = page.extract_tables()
            pdf_dict['tables'] = tables

            print(pdf_dict)
            result.append(pdf_dict)


pdf_path = r"D:\wk\deepLearning_LLMRagCompetition\data\2389de12d78fe1ca4fa24910e6b1573902098bc3.PDF"
read_pdf(pdf_path)



