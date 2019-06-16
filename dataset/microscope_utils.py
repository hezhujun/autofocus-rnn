class Microscope(object):

    def __init__(self, group, init_pos, transform=None):
        self.group = group
        self.pos_init = init_pos
        if init_pos < self.pos_min or init_pos > self.pos_max:
            raise ValueError("init pos is invalid")
        self.pos_cur = init_pos
        self.want_move_history = []
        self.move_history = []
        self.pos_history = []
        self.movement_out_of_right_count = 0
        self.movement_out_of_left_count = 0
        self.image_transform = transform

    def __repr__(self):
        return "Microscopy (%s) init pos %d" % (self.group.name, self.pos_init)

    @property
    def current_position(self):
        return self.group.positions[self.pos_cur]

    @property
    def current_image(self):
        image = self.group.positions[self.pos_cur].get_image()
        if self.image_transform:
            image = self.image_transform(image)
        return image

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
        idx_distance = round(distance / (self.units_per_min_step * 0.05))
        return int(idx_distance)

    def distance_to_peak(self):
        return float((self.pos_peak - self.pos_cur) * self.units_per_min_step * 0.05)

    def idx_distance_to_peak(self):
        return self.pos_peak - self.pos_cur

    def move(self, idx_distance):
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
