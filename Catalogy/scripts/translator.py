import pandas as pd
from googletrans import Translator
import asyncio

async def translate_text(translator, text, src, dest):
    return (await translator.translate(text, src=src, dest=dest)).text

async def translate_table(df, translator, src='fr', dest='en'):
    translated_df = df.copy()

    translated_columns = {}
    for col in df.columns:
        translated_col = await translate_text(translator, col, src, dest)
        translated_columns[col] = translated_col
    translated_df.rename(columns=translated_columns, inplace=True)

    for col in df.columns:
        for index, value in df[col].items():
            if isinstance(value, str):
                translated_df.at[index, translated_columns[col]] = await translate_text(
                    translator, value, src, dest
                )

    return translated_df

async def main():
    translator = Translator()

    input_file1 = '../Data/cats_data1.xlsx'
    sheet_name1 = 'Code'
    code_data = pd.read_excel(input_file1, sheet_name=sheet_name1)
    code_data_translated = await translate_table(code_data, translator)

    input_file2 = '../data_augmentation/cats_data_aug.xlsx'
    data_aug = pd.read_excel(input_file2)

    data_aug_translated = data_aug.copy()
    translated_columns = {
        col: await translate_text(translator, col, src='fr', dest='en') for col in data_aug.columns
    }
    data_aug_translated.rename(columns=translated_columns, inplace=True)

    output_file = '../Data/cats_data_en.xlsx'
    with pd.ExcelWriter(output_file) as writer:
        code_data_translated.to_excel(writer, sheet_name='Code', index=False)
        data_aug_translated.to_excel(writer, sheet_name='Data', index=False)

    print(f"Translated Excel file saved to {output_file}")

asyncio.run(main())
