import random
import pandas as pd


vild_template = [
    lambda c: f"a photo of a {c} in the scene",
    ]


# input_filename = '/home/jeffliang/zsseg/datasets/coco/meta/coco_nouns.csv'
# output_filename = '/home/jeffliang/zsseg/datasets/coco/meta/diverse_coco_nouns.csv'
# out = open(output_filename, 'a')
# df = pd.read_csv(input_filename, sep='\t')
# out.write("title\tfilepath\n")
#
# images = df['filepath'].tolist()
# captions = df['title'].tolist()
#
# # df_new = df.drop_duplicates(subset=['filepath'])
#
#
# for img, cap in zip(images, captions):
#     class_name = cap.split(' ')[-1][:-1]
#     template = random.choice(vild_template)
#     new_cap = template(class_name)
#     out.write("%s\t%s\n" % (new_cap, img))


input_filename = '/home/jeffliang/zsseg/datasets/coco/meta/coco_nouns.csv'
output_filename = '/home/jeffliang/zsseg/datasets/coco/meta/prompted_coco_nouns.csv'
out = open(output_filename, 'a')
df = pd.read_csv(input_filename, sep='\t')
out.write("title\tfilepath\n")

images = df['filepath'].tolist()
captions = df['title'].tolist()

# df_new = df.drop_duplicates(subset=['filepath'])


for img, cap in zip(images, captions):
    class_name = cap.split(' ')[-1][:-1]
    template = vild_template[0]
    out.write("%s\t%s\n" % (template(class_name), img))




# for key, value in df.groupby('filepath'):
#     words_list = value['title'].tolist()
#     try:
#         words_str = '|'.join(words_list)
#         out.write("%s\t%s\n" % (words_str, key))
#     except:
#         print(words_list)
#
# # images = df_new['filepath'].tolist()
# # captions = df_new['title'].tolist()
#
# # for img, cap in zip(images, captions):
# #     out.write("%s\t%s\n" % (cap, img))
# # df_new.to_csv('/home/jeffliang/zsseg/datasets/coco/coco_train_merge_captions.csv', index=False)
# # images = df['filepath'].tolist()
# # captions = df['title'].tolist()ate = [


# for key, value in df.groupby('filepath'):
#     words_list = value['title'].tolist()
#     try:
#         words_str = '|'.join(words_list)
#         out.write("%s\t%s\n" % (words_str, key))
#     except:
#         print(words_list)
#
# # images = df_new['filepath'].tolist()
# # captions = df_new['title'].tolist()
#
# # for img, cap in zip(images, captions):
# #     out.write("%s\t%s\n" % (cap, img))
# # df_new.to_csv('/home/jeffliang/zsseg/datasets/coco/coco_train_merge_captions.csv', index=False)
# # images = df['filepath'].tolist()
# # captions = df['title'].tolist()