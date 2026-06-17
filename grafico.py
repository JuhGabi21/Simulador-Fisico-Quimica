import tkinter as tk
from tkinter import ttk
import numpy as np
import random

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# =========================
# CONSTANTES
# =========================

k = 1.380649e-23

GASES = {
    "He": 4.0026,
    "Ne": 20.18,
    "Ar": 39.95,
    "Kr": 83.80,
    "Xe": 131.29
}


def lighten_color(hex_color, factor=0.3):
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(
        hex_color[2:4], 16), int(hex_color[4:6], 16)
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f'#{r:02x}{g:02x}{b:02x}'

# =========================
# CLASSE DO ÁTOMO
# =========================


class Atom:
    def __init__(self, canvas, width, height, temperature, speed_scale, size, mass, color, mode_3d=False):

        self.canvas = canvas
        self.width = width
        self.height = height
        self.base_size = size
        self.size = size
        self.mass = mass
        self.color = color
        self.mode_3d = mode_3d

        self.x = random.randint(20, width - 20)
        self.y = random.randint(20, height - 20)
        self.z = random.uniform(0.1, 1.0) if mode_3d else 0.5

        # Velocidade baseada na raiz da temperatura sobre a massa (dependência térmica correta)
        # O speed_scale atua apenas como um ganho estético/visual
        mag = random.uniform(0.5, 1.5)
        speed = mag * np.sqrt(temperature / mass) * (speed_scale * 0.1)

        if mode_3d:
            theta = random.uniform(0, 2 * np.pi)
            phi = np.arccos(random.uniform(-1, 1))
            self.vx = np.sin(phi) * np.cos(theta) * speed
            self.vy = np.sin(phi) * np.sin(theta) * speed
            self.vz = np.cos(phi) * speed
        else:
            angle = random.uniform(0, 2 * np.pi)
            self.vx = np.cos(angle) * speed
            self.vy = np.sin(angle) * speed
            self.vz = 0.0

        self.ids = []

        if mode_3d:
            self.ids.append(canvas.create_oval(
                0, 0, 0, 0, fill=color, outline=""))
            c2 = lighten_color(color, 0.4)
            self.ids.append(canvas.create_oval(
                0, 0, 0, 0, fill=c2, outline=""))
            c3 = lighten_color(color, 0.8)
            self.ids.append(canvas.create_oval(
                0, 0, 0, 0, fill=c3, outline=""))
        else:
            self.ids.append(canvas.create_oval(
                0, 0, 0, 0, fill=color, outline="white"))

        self.update_visual()

    def move(self, dt=1.0):

        self.x += self.vx * dt
        self.y += self.vy * dt

        if self.mode_3d:
            # Escalado para manter Z entre ~0.1 e 1.0
            self.z += self.vz * (dt * 0.01)

        if self.x <= self.size:
            self.x = self.size
            self.vx = abs(self.vx)
        elif self.x >= self.width - self.size:
            self.x = self.width - self.size
            self.vx = -abs(self.vx)

        if self.y <= self.size:
            self.y = self.size
            self.vy = abs(self.vy)
        elif self.y >= self.height - self.size:
            self.y = self.height - self.size
            self.vy = -abs(self.vy)

        if self.mode_3d:
            if self.z <= 0.1:
                self.z = 0.1
                self.vz = abs(self.vz)
            elif self.z >= 1.0:
                self.z = 1.0
                self.vz = -abs(self.vz)

    def update_visual(self):
        # Depth scaling
        factor = (0.5 + 0.5 * self.z) if self.mode_3d else 1.0
        self.size = max(2, self.base_size * factor)

        if self.mode_3d:
            size = self.size
            self.canvas.coords(
                self.ids[0], self.x - size, self.y - size, self.x + size, self.y + size)

            s2 = size * 0.7
            offs2 = size * 0.2
            self.canvas.coords(self.ids[1], self.x - s2 - offs2, self.y -
                               s2 - offs2, self.x + s2 - offs2, self.y + s2 - offs2)

            s3 = size * 0.3
            offs3 = size * 0.4
            self.canvas.coords(self.ids[2], self.x - s3 - offs3, self.y -
                               s3 - offs3, self.x + s3 - offs3, self.y + s3 - offs3)
        else:
            self.canvas.coords(
                self.ids[0],
                self.x - self.size,
                self.y - self.size,
                self.x + self.size,
                self.y + self.size
            )

    def speed(self):
        if self.mode_3d:
            return np.sqrt(self.vx**2 + self.vy**2 + self.vz**2)
        return np.sqrt(self.vx**2 + self.vy**2)

# =========================
# APLICAÇÃO
# =========================


class MaxwellSimulation:

    def __init__(self, root):

        self.root = root
        self.root.title("Simulador de Maxwell-Boltzmann")
        self.root.geometry("1400x850")
        self.root.configure(bg="#fafafa")

        self.running = False

        self.atom_count = tk.IntVar(value=0)
        self.atom_count.trace_add("write", self.update_atom_count_event)
        self.temperature = tk.IntVar(value=300)
        self.prev_temperature = 300
        self.atom_size = tk.IntVar(value=6)
        self.speed_scale = tk.IntVar(value=10)
        self.prev_speed_scale = 10
        self.mode_3d = tk.BooleanVar(value=False)
        self.dark_mode = tk.BooleanVar(value=False)

        self.selected_gases = {
            "He": tk.BooleanVar(value=True),
            "Ne": tk.BooleanVar(value=False),
            "Ar": tk.BooleanVar(value=False),
            "Kr": tk.BooleanVar(value=False),
            "Xe": tk.BooleanVar(value=False)
        }

        self.gas_colors = {
            "He": "#9b51e0",  # Roxo
            "Ne": "#e74c3c",  # Vermelho
            "Ar": "#3498db",  # Azul
            "Kr": "#f1c40f",  # Amarelo
            "Xe": "#2ecc71"   # Verde
        }

        self.atoms = []

        self.create_layout()
        self.root.update()
        self.create_atoms()
        self.update_graph()

    # =========================
    # INTERFACE
    # =========================

    def create_atoms(self):
        """Create atoms for the simulation"""
        if hasattr(self, 'canvas'):
            self.canvas.delete("all")

        self.atoms = []

        active_gases = [gas for gas,
                        var in self.selected_gases.items() if var.get()]
        if not active_gases:
            return

        total_atoms = self.atom_count.get()
        if total_atoms <= 0:
            return

        atoms_per_gas = total_atoms // len(active_gases)
        remainder = total_atoms % len(active_gases)

        counts = {gas: atoms_per_gas for gas in active_gases}
        for i in range(remainder):
            counts[active_gases[i]] += 1

        temp = self.temperature.get()

        for gas in active_gases:
            mass = GASES[gas]
            color = self.gas_colors[gas]

            for _ in range(counts[gas]):
                atom = Atom(
                    self.canvas,
                    self.canvas.winfo_width(),
                    self.canvas.winfo_height(),
                    temp,
                    self.speed_scale.get(),
                    self.atom_size.get(),
                    mass,
                    color,
                    self.mode_3d.get()
                )
                self.atoms.append(atom)

    def create_layout(self):

        # Cores base da imagem
        BG_MAIN = "#fafafa"
        PURPLE = "#9b51e0"
        GREEN_TEXT = "#2ecc71"
        GRAY_BTN = "#b2bec3"

        # =========================
        # MAIN FRAME
        # =========================
        main_frame = tk.Frame(self.root, bg=BG_MAIN)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # LADO ESQUERDO (Botoões + Canvas + Slider)
        left_frame = tk.Frame(main_frame, bg=BG_MAIN)
        left_frame.pack(side="left", fill="both", expand=True)

        # Botoões
        top_buttons = tk.Frame(left_frame, bg=BG_MAIN)
        top_buttons.pack(fill="x", pady=(0, 20))

        self.play_button = tk.Button(
            top_buttons,
            text="▶||",
            command=self.toggle_simulation,  # type: ignore
            bg=PURPLE,
            fg=GREEN_TEXT,
            font=("Arial", 16, "bold"),
            relief="flat",
            bd=0,
            padx=10
        )
        self.play_button.pack(side="left")

        reset_button = tk.Button(
            top_buttons,
            text="Zerar tudo",
            command=self.reset_simulation,  # type: ignore
            bg="#b2bec3",
            fg=GREEN_TEXT,
            font=("Arial", 12, "bold"),
            relief="flat",
            bd=0,
            padx=15,
            pady=5
        )
        reset_button.pack(side="left", padx=20)

        self.toggle_3d_button = tk.Button(
            top_buttons,
            text="Modo 2D",
            command=self.toggle_3d_mode,
            bg="#b2bec3",
            fg=GREEN_TEXT,
            font=("Arial", 12, "bold"),
            relief="flat",
            bd=0,
            padx=15,
            pady=5
        )
        self.toggle_3d_button.pack(side="left", padx=5)

        # Canvas Frame com borda roxa
        self.canvas_frame = tk.Frame(left_frame, bg=PURPLE, bd=2)
        self.canvas_frame.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(
            self.canvas_frame, bg=BG_MAIN, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=2, pady=2)

        # Slider de tamanho
        size_frame = tk.Frame(left_frame, bg=BG_MAIN)
        size_frame.pack(fill="x", pady=10)

        tk.Label(size_frame, text="Ajustar o tamanho dos",
                 bg=BG_MAIN, fg=GREEN_TEXT, font=("Arial", 10)).pack()
        tk.Label(size_frame, text="átomos", bg=BG_MAIN,
                 fg=GREEN_TEXT, font=("Arial", 10)).pack()

        tk.Scale(
            size_frame, from_=2, to=20, orient="horizontal", variable=self.atom_size,
            command=self.update_atom_size, length=300, bg=BG_MAIN, fg=PURPLE,
            troughcolor=BG_MAIN, highlightthickness=0, showvalue=False
        ).pack()

        # Botão Modo Escuro no canto inferior esquerdo
        self.toggle_dark_button = tk.Button(
            left_frame,
            text="🌙",
            command=self.toggle_dark_mode,
            bg=GRAY_BTN,
            fg=GREEN_TEXT,
            font=("Arial", 9),
            relief="flat",
            bd=0,
            padx=5,
            pady=2
        )
        self.toggle_dark_button.pack(side="bottom", anchor="sw")

        # =========================
        # LADO DIREITO
        # =========================
        right_frame = tk.Frame(main_frame, bg=BG_MAIN)
        right_frame.pack(side="right", fill="both", padx=(20, 0))

        # Controles Topo Direito
        controls_top = tk.Frame(right_frame, bg=BG_MAIN)
        controls_top.pack(fill="x")

        # Coluna Quantidade e Temperatura
        qt_frame = tk.Frame(controls_top, bg=BG_MAIN)
        qt_frame.pack(side="left", fill="y", padx=(0, 20))

        # Qtd átomos
        qt_box = tk.Frame(qt_frame, bg=PURPLE, bd=2)
        qt_box.pack(fill="x", pady=(0, 20))
        qt_inner = tk.Frame(qt_box, bg=BG_MAIN)
        qt_inner.pack(fill="both", expand=True, padx=2, pady=2)

        tk.Label(qt_inner, text="Quantidade de átomos:", bg=BG_MAIN,
                 fg=GREEN_TEXT, font=("Arial", 10)).pack(pady=5)
        tk.Entry(qt_inner, textvariable=self.atom_count, justify="center", bg=BG_MAIN,
                 fg=PURPLE, font=("Arial", 10), relief="groove", bd=1).pack(pady=(0, 5), padx=10)

        # Temperatura
        tk.Label(qt_frame, text="Escolha a temperatura:", bg=BG_MAIN,
                 fg=GREEN_TEXT, font=("Arial", 10)).pack()
        tk.Scale(
            qt_frame, from_=100, to=1000, resolution=100, orient="horizontal", variable=self.temperature,
            command=self.update_temperature, length=200, bg=BG_MAIN, fg=PURPLE,
            troughcolor=BG_MAIN, highlightthickness=0
        ).pack()

        # Grade Gases
        gas_box = tk.Frame(controls_top, bg=PURPLE, bd=2)
        gas_box.pack(side="right", fill="y")
        gas_inner = tk.Frame(gas_box, bg=BG_MAIN)
        gas_inner.pack(fill="both", expand=True, padx=2, pady=2)

        tk.Label(gas_inner, text="Escolha o Gás:", bg=BG_MAIN,
                 fg=GREEN_TEXT, font=("Arial", 11)).pack(pady=5)

        gas_grid = tk.Frame(gas_inner, bg=BG_MAIN)
        gas_grid.pack(padx=10, pady=10)

        row, col = 0, 0
        for gas in GASES.keys():
            tk.Checkbutton(
                gas_grid, text=gas, variable=self.selected_gases[gas], command=self.change_gas,
                bg=BG_MAIN, fg=self.gas_colors[gas], font=("Arial", 12, "bold"), selectcolor=BG_MAIN,
                activebackground=BG_MAIN
            ).grid(row=row, column=col, padx=10, pady=5)
            row += 1
            if row > 2:
                row = 0
                col += 1

        # Gráfico Lado Direito
        self.plot_frame_border = tk.Frame(right_frame, bg=PURPLE, bd=2)
        self.plot_frame_border.pack(fill="both", expand=True, pady=(20, 10))

        self.fig = Figure(figsize=(5, 4), dpi=100, facecolor=BG_MAIN)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(BG_MAIN)
        self.ax.tick_params(colors=GREEN_TEXT)
        self.ax.spines['bottom'].set_color(PURPLE)
        self.ax.spines['top'].set_color(PURPLE)
        self.ax.spines['right'].set_color(PURPLE)
        self.ax.spines['left'].set_color(PURPLE)

        self.canvas_plot = FigureCanvasTkAgg(
            self.fig, master=self.plot_frame_border)
        self.canvas_plot.get_tk_widget().pack(fill="both", expand=True, padx=2, pady=2)

        # Scale velocidade
        speed_frame = tk.Frame(right_frame, bg=BG_MAIN)
        speed_frame.pack(fill="x")

        tk.Label(speed_frame, text="Escala de velocidade", bg=BG_MAIN,
                 fg=GREEN_TEXT, font=("Arial", 10)).pack(side="right")
        tk.Scale(
            speed_frame, from_=5, to=50, orient="horizontal", variable=self.speed_scale,
            command=self.update_speed_scale, length=200, bg=BG_MAIN, fg=PURPLE,
            troughcolor=BG_MAIN, highlightthickness=0, showvalue=False
        ).pack(side="right", padx=10)

    def toggle_simulation(self):

        self.running = not self.running

        if self.running:
            self.play_button.config(text="⏸ Pause")
            self.animate()
        else:
            self.play_button.config(text="▶ Play")

    # =========================
    # RESET
    # =========================

    def reset_simulation(self):

        self.running = False
        self.play_button.config(text="▶ Play")

        for gas, var in self.selected_gases.items():
            var.set(gas == "He")

        self.temperature.set(300)
        self.prev_temperature = 300
        self.atom_count.set(0)
        self.atom_size.set(6)
        self.speed_scale.set(10)
        self.prev_speed_scale = 10

        self.create_atoms()
        self.update_graph()

    # =========================
    # QUANTIDADE DE ÁTOMOS
    # =========================

    def update_atom_count_event(self, *args):
        try:
            val = self.atom_count.get()
            if val > 0:
                self.create_atoms()
                self.update_graph()
        except:
            pass

    # =========================
    # TEMPERATURA
    # =========================

    def update_temperature(self, event=None):

        temp = self.temperature.get()

        if not hasattr(self, 'prev_temperature'):
            self.prev_temperature = 300

        factor = np.sqrt(temp / self.prev_temperature)

        for atom in self.atoms:
            atom.vx *= factor
            atom.vy *= factor

        self.prev_temperature = temp
        self.update_graph()

    # =========================
    # TROCAR GÁS
    # =========================

    def change_gas(self):
        self.create_atoms()
        self.update_graph()

    # =========================
    # MODO 3D
    # =========================

    def toggle_3d_mode(self):
        self.mode_3d.set(not self.mode_3d.get())
        if self.mode_3d.get():
            self.toggle_3d_button.config(
                text="Modo 3D", bg="#3498db", fg="white")
        else:
            self.toggle_3d_button.config(
                text="Modo 2D", bg="#b2bec3", fg="#2ecc71")
        self.create_atoms()
        self.update_graph()

    # =========================
    # MODO ESCURO
    # =========================

    def toggle_dark_mode(self):
        self.dark_mode.set(not self.dark_mode.get())

        if self.dark_mode.get():
            self.toggle_dark_button.config(
                text="☀️", bg="#f1c40f", fg="#2b2b2b")
            bg_color = "#1e1e1e"
        else:
            self.toggle_dark_button.config(
                text="🌙", bg="#b2bec3", fg="#2ecc71")
            bg_color = "#fafafa"

        def update_colors(widget):
            try:
                # Altera apenas widgets que tem as cores base de fundo
                if widget.cget("bg") in ["#fafafa", "#1e1e1e"]:
                    widget.configure(bg=bg_color)
                if widget.winfo_class() == "Scale":
                    widget.configure(troughcolor=bg_color)
                if widget.winfo_class() == "Checkbutton":
                    widget.configure(selectcolor=bg_color,
                                     activebackground=bg_color)
            except:
                pass
            for child in widget.winfo_children():
                update_colors(child)

        update_colors(self.root)

        # Atualiza gráfico
        self.fig.patch.set_facecolor(bg_color)
        self.ax.set_facecolor(bg_color)
        self.canvas_plot.draw()

    # =========================
    # TAMANHO DOS ÁTOMOS
    # =========================

    def update_atom_size(self, event=None):

        size = self.atom_size.get()

        for atom in self.atoms:
            atom.base_size = size

        self.update_graph()

    # =========================
    # ESCALA VELOCIDADE
    # =========================

    def update_speed_scale(self, event=None):

        scale = self.speed_scale.get()

        if not hasattr(self, 'prev_speed_scale'):
            self.prev_speed_scale = 10

        factor = scale / self.prev_speed_scale

        for atom in self.atoms:
            atom.vx *= factor
            atom.vy *= factor

        self.prev_speed_scale = scale
        self.update_graph()

    # =========================
    # GRÁFICO E ANIMAÇÃO
    # =========================

    def check_collisions(self):
        # Implementação de Grade Espacial (Spatial Grid) para O(N)
        # Garante um grid mínimo para evitar divisões excessivas
        grid_size = float(max(20, self.atom_size.get() * 4))
        cells = {}

        # Alocar os átomos nas células da grade
        for atom in self.atoms:
            grid_x = int(atom.x / grid_size)
            grid_y = int(atom.y / grid_size)
            # A colisão em 3D usa o plano projetado na tela para garantir que
            # as bolas que se sobrepõem visualmente realmente batam.
            grid_z = 0

            key = (grid_x, grid_y, grid_z)
            if key not in cells:
                cells[key] = []
            cells[key].append(atom)

        checked = set()

        # Checar colisões apenas com átomos da mesma célula e adjacentes
        z_range = [0]

        for key, cell_atoms in cells.items():
            gx, gy, gz = key
            adjacent_atoms = []

            for dx_ in [-1, 0, 1]:
                for dy_ in [-1, 0, 1]:
                    adj_key = (gx + dx_, gy + dy_, gz)
                    if adj_key in cells:
                        adjacent_atoms.extend(cells[adj_key])

            for a1 in cell_atoms:
                for a2 in adjacent_atoms:
                    if a1 is a2:
                        continue

                    # Identificador único do par
                    pair_id = tuple(sorted([id(a1), id(a2)]))
                    if pair_id in checked:
                        continue
                    checked.add(pair_id)

                    dx = a1.x - a2.x
                    dy = a1.y - a2.y

                    dist_sq = dx**2 + dy**2
                    min_dist = a1.size + a2.size

                    if dist_sq <= min_dist**2:
                        dist = np.sqrt(dist_sq) if dist_sq > 0 else min_dist

                        if dist_sq == 0:
                            nx, ny = 1.0, 0.0
                        else:
                            nx = dx / dist
                            ny = dy / dist

                        dvx = a1.vx - a2.vx
                        dvy = a1.vy - a2.vy

                        m1 = a1.mass
                        m2 = a2.mass
                        dot_product = (dvx * nx + dvy * ny)
                        p = 2.0 * dot_product / (m1 + m2)

                        if dot_product < 0:
                            a1.vx -= p * m2 * nx
                            a1.vy -= p * m2 * ny
                            a2.vx += p * m1 * nx
                            a2.vy += p * m1 * ny

                        overlap = 0.5 * (min_dist - dist) + 0.1
                        a1.x += nx * overlap
                        a1.y += ny * overlap
                        a2.x -= nx * overlap
                        a2.y -= ny * overlap

    def animate(self):
        if self.running:
            max_speed = max((atom.speed() for atom in self.atoms), default=0.0)
            desired_step = max(2.0, self.atom_size.get() * 0.5)
            sub_steps = int(np.clip(np.ceil(max_speed / desired_step), 5, 20))
            dt = 1.0 / sub_steps

            for _ in range(sub_steps):
                for atom in self.atoms:
                    atom.move(dt)
                self.check_collisions()

            # Atualiza visual na tela apenas 1x por frame
            for atom in self.atoms:
                atom.update_visual()

            self.root.after(30, self.animate)

    def update_graph(self, event=None):

        self.ax.clear()

        try:
            T = self.temperature.get()
            N = self.atom_count.get()
            active_gases = [gas for gas,
                            var in self.selected_gases.items() if var.get()]
        except:
            return

        if not active_gases:
            self.canvas_plot.draw()
            return

        N_per_gas = N / len(active_gases) if N > 0 else 1

        # Achar limite do eixo x usando o mais leve gás selecionado e T=1000K
        min_mass_amu = min([GASES[g] for g in active_gases])

        # Ajustado cálculo do range do X_axis
        v_rms_max = np.sqrt(3 * k * 1000 / (min_mass_amu * 1.660539e-27))
        v = np.linspace(0, v_rms_max * 2.5, 500)

        temps_to_plot = sorted(list(set([100, T, 1000])))

        for gas in active_gases:
            mass_amu = GASES[gas]
            m = mass_amu * 1.660539e-27
            color = self.gas_colors[gas]

            for temp in temps_to_plot:

                if self.mode_3d.get():
                    # 3D: Maxwell-Boltzmann 3D real
                    # f(v) = 4 * pi * (m / 2pi*kT)^(3/2) * v^2 * exp(-mv^2 / 2kT)
                    coef = 4 * np.pi * ((m / (2 * np.pi * k * temp)) ** 1.5)
                    f_v = coef * (v ** 2) * np.exp(-m *
                                                   (v ** 2) / (2 * k * temp))
                    v_p = np.sqrt(2 * k * temp / m)
                else:
                    # 2D: Maxwell-Boltzmann 2D real
                    # f(v) = (m / kT) * v * exp(-mv^2 / 2kT)
                    f_v = (m / (k * temp)) * v * \
                        np.exp(-m * (v ** 2) / (2 * k * temp))
                    v_p = np.sqrt(k * temp / m)

                is_current = (temp == T)
                lw = 2.5 if is_current else 1.0
                alpha = 0.9 if is_current else 0.3
                ls = '-' if is_current else '--'

                self.ax.plot(v, f_v, color=color, linewidth=lw,
                             alpha=alpha, linestyle=ls)

                peak_idx = np.abs(v - v_p).argmin()
                label_text = f"{gas} ({temp}K)" if is_current else f"{temp}K"

                self.ax.text(v_p, f_v[peak_idx] * 1.05, label_text, color=color,
                             ha='center', fontsize=9, fontweight='bold', alpha=alpha)

        self.ax.set_xlabel("Velocidade (m/s)", color="#2ecc71")
        self.ax.set_ylabel("Quantidade de átomos", color="#2ecc71")
        self.ax.tick_params(colors='#9b51e0')
        self.ax.set_ylim(bottom=0)
        self.ax.set_xlim(left=0)

        # Atualizar a plotagem
        self.canvas_plot.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = MaxwellSimulation(root)
    root.mainloop()
