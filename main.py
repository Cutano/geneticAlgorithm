import random
import math
import bisect
import numpy as np
from PIL import Image, ImageDraw


#  目标背景最好为纯色 在下方指定背景色


class Color:
    def __init__(self, rgba):
        self.rgba = rgba

    def get_rgba(self):
        return tuple(self.rgba)

    def get_rgb(self):
        return tuple(self.rgba[:-1])


back_color = Color([255, 255, 255, 255])  # 背景色（重要）
triangle_amount = 40  # 单张图片内三角形个数，尽量是4的倍数
obsolete_rate = 0.6  # 淘汰率
mutation_rate = 0.35  # 个体变异率
dna_mutation_possibility = 0.2  # 准变异个体中每段DNA突变率
mutation_level = 0.25  # 变异幅度（偏移量）


# 单个三角形
class Triangle:
    def __init__(self, vertexes, color):
        self.vertexes = vertexes
        self.color = color
        self.dna = []
        for vertex in self.vertexes:
            self.dna.append(vertex[0])
            self.dna.append(vertex[1])
        for i in range(4):
            self.dna.append(self.color.rgba[i])


# 单个图片个体类
class Picture:
    def __init__(self, dna=None, tri_set=None, tar=None):
        if dna is None and tri_set is None:  # 随机生成一张图
            self.tri_set = [gen_rand_tri(tar, tar.size) for i in range(triangle_amount)]
            self.dna = []
            self.gen_dna()
        elif dna is None:  # 从给定三角形列表生成
            self.dna = []
            self.tri_set = tri_set
            self.gen_dna()
        elif tri_set is None:  # 从给定DNA生成
            self.tri_set = []
            self.dna = dna
            self.gen_tris_from_dna()

    # 生成DNA
    def gen_dna(self):
        for tri in self.tri_set:
            self.dna += tri.dna

    # 根据DNA生成三角形列表
    def gen_tris_from_dna(self):
        for i in range(0, len(self.dna), 10):
            vertexes = (
                (self.dna[i], self.dna[i + 1]), (self.dna[i + 2], self.dna[i + 3]), (self.dna[i + 4], self.dna[i + 5]))
            color = Color([self.dna[i + 6], self.dna[i + 7], self.dna[i + 8], self.dna[i + 9]])
            self.tri_set.append(Triangle(vertexes, color))

    # 画出当前图片
    def draw(self, tar):
        can = Image.new("RGBA", tar.size, back_color.get_rgb())
        for i in self.tri_set:
            new = Image.new("RGBA", tar.size)
            drawer = ImageDraw.Draw(new)
            drawer.polygon(i.vertexes, fill=i.color.get_rgba())
            can = Image.alpha_composite(can, new)
        return can

    # 获取不适应度
    def get_fitness(self, tar, mat_tar):
        mat_curr = np.asarray(self.draw(tar), dtype=np.int32)
        return np.linalg.norm(abs(mat_tar - mat_curr))  # 以向量距离衡量差距

    # 变异
    def mutate(self, ratio, level, size):
        for i in range(0, len(self.dna), 10):
            if random.random() < ratio:
                index = i + random.randint(0, 9)
                if index - i < 6:  # 坐标变异
                    if (index - 1) % 2 == 1:  # 垂直坐标变异
                        bot = min(size[1], self.dna[index] + level * size[1])
                        top = max(0, self.dna[index] - level * size[1])
                        self.dna[index] = random.randint(int(top), int(bot))
                    else:  # 水平坐标变异
                        left = max(0, self.dna[index] - level * size[0])
                        right = min(size[0], self.dna[index] + level * size[0])
                        self.dna[index] = random.randint(int(left), int(right))
                else:  # 颜色变异
                    top = min(255, int(self.dna[index] + level * 255))
                    bot = max(0, int(self.dna[index] - level * 255))
                    self.dna[index] = random.randint(int(bot), int(top))


# 整体种群控制器
class Controller:
    def __init__(self, amount, generations, targetImg):
        self.amount = amount
        self.generations = generations
        self.pics = [Picture(tar=targetImg) for i in range(amount)]
        self.mat_tar = np.asarray(targetImg, dtype=np.int32)
        print("stand by...")

    # 进行迭代
    def generate(self, auto_save=0):
        for i in range(1, self.generations + 1):
            print(f"generating {i} generation...")
            fit_value = np.array([pic.get_fitness(targetImg, self.mat_tar) for pic in self.pics])
            chosen = ga_selection(fit_value, maxObsolete=int(self.amount * obsolete_rate))  # 将劣势个体淘汰掉
            self.pics = [self.pics[index] for index in chosen]
            new_pics = []
            while len(new_pics) != self.amount:  # 杂交生成下一代
                new_pic = self.cross(self.pics[random.randint(0, len(self.pics) - 1)], self.pics[random.randint(0, len(self.pics) - 1)])
                if random.random() < mutation_rate:
                    new_pic.mutate(dna_mutation_possibility, mutation_level, targetImg.size)
                new_pics.append(new_pic)
            self.pics = new_pics
            if auto_save != 0:  # 自动保存
                if i % auto_save == 0:
                    self.draw_best().save(f"./result/gen{i}.png", "png")

    # 杂交
    def cross(self, pic_1, pic_2):
        dna = []
        dna += pic_1.dna[:len(pic_1.dna)]
        dna += pic_2.dna[len(pic_2.dna):]
        return Picture(dna=dna)

    # 获取当代最优个体
    def draw_best(self):
        best_ele = np.array([pic.get_fitness(targetImg, self.mat_tar) for pic in self.pics]).argmin()
        print(self.pics[best_ele].get_fitness(targetImg, self.mat_tar))
        return self.pics[best_ele].draw(targetImg)


# 轮盘赌算法
def ga_selection(fit_value, maxObsolete):
    # 不适应度总和
    fit_value -= fit_value.min()
    fit_value *= 7.5  # 扩大斜率 增加差异度
    total_fit = fit_value.sum()
    fit_value_ratio = np.array([i / total_fit for i in fit_value])  # 计算每个不适应度占适应度总和的比例
    # 计算累计概率
    for i in range(1, len(fit_value_ratio)):
        fit_value_ratio[i] += fit_value_ratio[i - 1]
    obsoleted = set(bisect.bisect(fit_value_ratio, random.random()) for i in range(maxObsolete))  # 获取被淘汰个体索引
    chosen = set()
    for i in range(len(fit_value_ratio)):
        if i not in obsoleted:
            chosen.add(i)
    return chosen


# 生成三个顶点
def gen_vertexes(size):
    a = (random.random() * size[0], random.random() * size[1])
    b = (random.random() * size[0], random.random() * size[1])
    c = (random.random() * size[0], random.random() * size[1])
    return a, b, c


# 生成颜色
def gen_color(tar, vertexes):
    center = (
        (vertexes[0][0] + vertexes[1][0] + vertexes[2][0]) / 3, (vertexes[0][1] + vertexes[1][1] + vertexes[2][1]) / 3)
    pixel = tar.getpixel(center)  # 以三角形的重心的颜色初始化三角形
    color = Color(list(pixel))
    color.rgba[3] = int(random.random() * 255)
    return color


# 生成与背景色不同的随机三角形
def gen_rand_tri(tar, size):
    vertexes = gen_vertexes(size)  # 获取随机顶点
    color = gen_color(tar, vertexes)  # 获取随机颜色
    while color.get_rgb() == back_color.get_rgb():  # 排除与背景色相同的颜色
        vertexes = gen_vertexes(size)
        color = gen_color(tar, vertexes)
    triangle = Triangle(vertexes, color)
    return triangle


if __name__ == '__main__':
    targetImg = Image.open("edge_small.png")
    targetImg = targetImg.convert("RGBA")
    controller = Controller(20, 110000, targetImg)  # 种群大小，最大迭代，目标图
    controller.generate(auto_save=1000)  # 每几张图片自动保存一次
    # controller.draw_best().save("./result/final.png", "png")
