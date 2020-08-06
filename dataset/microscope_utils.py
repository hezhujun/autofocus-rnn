class Microscope(object):

    def __init__(self, group, init_pos_idx, transform=None, unit_distance=0.5):
        self.group = group
        self.pos_init = init_pos_idx
        if init_pos_idx < self.pos_min or init_pos_idx > self.pos_max:
            raise ValueError("init pos idx is invalid")
        self.pos_cur = init_pos_idx
        self.want_move_history = []
        self.move_history = []
        self.pos_history = []
        self.movement_out_of_right_count = 0
        self.movement_out_of_left_count = 0
        self.image_transform = transform
        self.unit_distance = unit_distance  # distance per unit in z-axis

        self.pos_idx_to_position = {pos.pos_idx:pos for pos in group.positions}

    def __repr__(self):
        return "Microscopy (%s) init pos idx %d" % (self.group.name, self.pos_init)

    @property
    def current_position(self):
        return self.pos_idx_to_position[self.pos_cur]

    @property
    def current_image(self):
        image = self.pos_idx_to_position[self.pos_cur].get_image(self.image_transform)
        return image

    def get_current_focus_measure(self, key):
        return self.pos_idx_to_position[self.pos_cur].focus_measures[key]

    @property
    def is_in_right(self):
        return self.pos_cur == self.pos_max

    @property
    def is_in_focus(self):
        return self.pos_cur == self.pos_peak

    @property
    def pos_peak(self):
        return self.group.pos_peak_idx

    @property
    def pos_min(self):
        return self.group.positions[0].pos_idx

    @property
    def pos_max(self):
        return self.group.positions[-1].pos_idx

    @property
    def units_per_min_step(self):
        return abs(self.group.positions[1].z_coordinate - self.group.positions[0].z_coordinate)

    def convert_distance_to_idx_distance(self, distance):
        idx_distance = round(distance / self.unit_distance)
        return int(idx_distance)

    def distance_to_peak(self):
        return self.idx_distance_to_peak() * self.unit_distance

    def idx_distance_to_peak(self):
        return self.pos_peak - self.pos_cur

    def move(self, distance):
        idx_distance = int(distance)
        self.want_move_history.append(idx_distance)
        if idx_distance == 0:
            self.move_history.append(0)
            self.pos_history.append(self.pos_cur)
            return
        elif idx_distance < 0:
            if self.pos_cur + idx_distance >= self.pos_min:
                self.pos_cur = self.pos_cur + idx_distance
            else:
                # print("Error: self.pos_cur + idx_distance < self.pos_min")
                self.movement_out_of_left_count += 1
                idx_distance = self.pos_min - self.pos_cur
                self.pos_cur = self.pos_min
        elif idx_distance > 0:
            if self.pos_cur + idx_distance <= self.pos_max:
                self.pos_cur = self.pos_cur + idx_distance
            else:
                # print("Error: self.pos_cur + idx_distance > self.pos_min")
                self.movement_out_of_right_count += 1
                idx_distance = self.pos_max - self.pos_cur
                self.pos_cur = self.pos_max

        self.move_history.append(idx_distance)
        self.pos_history.append(self.pos_cur)

    def history(self):
        print("group {} peak pos {}".format(self.group.name, self.group.pos_peak_idx), end=" ")
        print("init pos %02d" % self.pos_init, end=" ")
        for i in range(len(self.move_history)):
            print("want to move %02d actually move %d to pos %02d" % (
                self.want_move_history[i], self.move_history[i], self.pos_history[i]), end=" ")
        print()


# from dataset.group_utils import *
# if __name__ == '__main__':
#     group = load_group_json("/root/userfolder/datasets/autofocus2/info/1.json", "/root/userfolder/datasets/autofocus2")
#     group.positions = group.positions[30:-20]
#     m = Microscope(group, 30)
#     assert m.current_position == group.positions[0]
#     assert m.is_in_right == False
#     assert m.is_in_focus == False
#     assert m.pos_peak == group.pos_peak_idx
#     assert m.pos_min == 30
#     assert m.pos_max == 79
#     assert m.idx_distance_to_peak() == group.pos_peak_idx - 30
#     m.move(group.pos_peak_idx - 30)
#     assert m.current_position == group.positions[group.pos_peak_idx - 30]
#     assert m.is_in_right == False
#     assert m.is_in_focus == True
#     assert m.pos_peak == group.pos_peak_idx
#     assert m.pos_min == 30
#     assert m.pos_max == 79
#     assert m.idx_distance_to_peak() == 0
#     m.move(79 - group.pos_peak_idx)
#     assert m.current_position == group.positions[-1]
#     assert m.is_in_right == True
#     assert m.is_in_focus == False
#     assert m.pos_peak == group.pos_peak_idx
#     assert m.pos_min == 30
#     assert m.pos_max == 79
#     assert m.idx_distance_to_peak() == group.pos_peak_idx - 79
#     m.history()