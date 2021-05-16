import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import mtree
import random
from time import time
from contextlib import contextmanager


db = None

root = tk.Tk()
root.title("MTree")
root.geometry("350x550")


def update_ui():
    for wd in (
        insert_entry,
        range_query,
        range_radius,
        knn_query,
        knn_k,
    ):
        if db is None:
            wd.configure(state=tk.DISABLED)
        else:
            wd.configure(state=tk.NORMAL)
    
    if db is not None:
        dimension_var.set(f"Vector dimension: {db.dimensions}")
        node_size_var.set(f"Node size: {db.node_size}")
        db_size_var.set(f"Tree size: {db.size}")


@contextmanager
def timeit():
    start = time()
    yield
    end = time()
    messagebox.showinfo(message=f"Operation took {end - start} seconds.\nDistance function has been called {db.dcall_counter} times.")


def validate_vector(string):
    try:
        inp = tuple(map(float, string.split(',')))
    except:
        messagebox.showerror(
            message="Invalid vector, type in scalars divided by ',' (example: '1,2,3,4,5' with dimension 5)"
        )
        return None
    
    if len(inp) != db.dimensions:
        messagebox.showerror(
            message="Invalid dimension of vector"
        )
        return None

    return inp


def insert_cb():
    string = insert_entry.get()

    inp = validate_vector(string)

    if inp:
        with timeit():
            db.add(inp)
    
    update_ui()


def range_mtree_cb():
    q_str = range_query.get()
    query_vector = validate_vector(q_str)
    if query_vector is None: return

    try:
        radius = int(range_radius.get())
    except:
        messagebox.showerror(message="Invalid radius")
    
    with timeit():
        res = db.range_search(query_vector, radius)
    
    result_output.delete(1.0, tk.END)
    result_output.insert(tk.INSERT, str(res))


def range_sequential_cb():
    q_str = range_query.get()
    query_vector = validate_vector(q_str)
    if query_vector is None: return

    try:
        radius = int(range_radius.get())
    except:
        messagebox.showerror(message="Invalid radius")
    
    with timeit():
        res = db.sequential_range_search(query_vector, radius)
    
    result_output.delete(1.0, tk.END)
    result_output.insert(tk.INSERT, str(res))


def knn_mtree_cb():
    q_str = knn_query.get()
    query_vector = validate_vector(q_str)
    if query_vector is None: return

    try:
        k = int(knn_k.get())
    except:
        messagebox.showerror(message="Invalid k")
    
    with timeit():
        res = set(db.knn(query_vector, k))
    
    result_output.delete(1.0, tk.END)
    result_output.insert(tk.INSERT, str(res))


def knn_sequential_cb():
    q_str = knn_query.get()
    query_vector = validate_vector(q_str)
    if query_vector is None: return

    try:
        k = int(knn_k.get())
    except:
        messagebox.showerror(message="Invalid k")
    
    with timeit():
        res = set(db.sequential_knn(query_vector, k))
    
    result_output.delete(1.0, tk.END)
    result_output.insert(tk.INSERT, str(res))


def generate_cb():
    def gen_db():
        global db
        with timeit():
            try:
                dimension = int(dimension_wd.get())
                count = int(vector_count_wd.get())
                db = mtree.MTree(node_size=int(node_size_wd.get()), dimensions=dimension)
            except ValueError as e:
                return messagebox.showerror(message=str(e))

            inp = [tuple(random.sample(range(0, 100), dimension)) for i in range(count)]

            db.add_bulk(inp)
            update_ui()

        insert_entry.insert(0, ", ".join(map(str, range(dimension))))
        generate_window.destroy()

    generate_window = tk.Toplevel(root)

    tk.Label(generate_window, text="Node size").pack()
    node_size_wd = tk.Entry(generate_window)
    node_size_wd.insert(0, "10")
    node_size_wd.pack()

    tk.Label(generate_window, text="Vector dimension").pack()
    dimension_wd = tk.Entry(generate_window)
    dimension_wd.insert(0, "2")
    dimension_wd.pack()

    tk.Label(generate_window, text="Number of vectors").pack()
    vector_count_wd = tk.Entry(generate_window)
    vector_count_wd.insert(0, 50)
    vector_count_wd.pack()

    tk.Label(generate_window, text="Lower range").pack()
    lower_range_wd = tk.Entry(generate_window)
    lower_range_wd.insert(0, 0)
    lower_range_wd.pack()

    tk.Label(generate_window, text="Upper range").pack()
    upper_range_wd = tk.Entry(generate_window)
    upper_range_wd.insert(0, 100)
    upper_range_wd.pack()

    tk.Button(generate_window, text="Generate", command=gen_db).pack()


def load_cb():
    global db

    path = tk.filedialog.askopenfilename(
        initialdir=".",
        title="Select mtree file",
    )

    try:
        if path:
            db = mtree.persistence.load_tree(path)
    except mtree.persistence.LoadError:
        messagebox.showerror(message="Invalid mtree file.")

    update_ui()


def save_cb():
    path = filedialog.asksaveasfilename(
        initialdir=".",
        title="Select destination",
    )

    if path:
        mtree.persistence.save_tree(db, path)


def create_menu(root):
    menu = tk.Menu(root)
    menu.add_command(label="Generate", command=generate_cb)
    menu.add_command(label="Load", command=load_cb)
    menu.add_command(label="Save", command=save_cb)
    root.config(menu=menu)


info_frame = tk.Frame(root, borderwidth=1, relief=tk.RAISED)
info_frame.pack(fill=tk.BOTH, expand=True)

dimension_var = tk.StringVar(value="Vector dimension: None")
tk.Label(info_frame, textvariable=dimension_var).pack()

node_size_var = tk.StringVar(value="Node size: None")
tk.Label(info_frame, textvariable=node_size_var).pack()

db_size_var = tk.StringVar(value="Tree size: None")
tk.Label(info_frame, textvariable=db_size_var).pack()

insert_frame = tk.Frame(root, borderwidth=1, relief=tk.RAISED)
insert_frame.pack(fill=tk.BOTH, expand=True)

tk.Label(insert_frame, text="Insert").pack()
insert_entry = tk.Entry(insert_frame)
insert_entry.pack()
tk.Button(insert_frame, text="Insert", command=insert_cb).pack(padx=4, pady=4)

range_frame = tk.Frame(root, borderwidth=1, relief=tk.RAISED)
range_frame.pack(fill=tk.BOTH, expand=True)

tk.Label(range_frame, text="Range search").pack()

tk.Label(range_frame, text="Query").pack()
range_query = tk.Entry(range_frame)
range_query.pack()

tk.Label(range_frame, text="Radius").pack()
range_radius = tk.Entry(range_frame)
range_radius.pack()

range_buttons_frame = tk.Frame(range_frame)
range_buttons_frame.pack()

tk.Button(range_buttons_frame, text="Mtree", command=range_mtree_cb).pack(side=tk.LEFT, padx=4, pady=4)
tk.Button(range_buttons_frame, text="Sequential", command=range_sequential_cb).pack(side=tk.RIGHT, padx=4, pady=4)

knn_frame = tk.Frame(root, borderwidth=1, relief=tk.RAISED)
knn_frame.pack(fill=tk.BOTH, expand=True)

tk.Label(knn_frame, text="kNN search").pack()

tk.Label(knn_frame, text="Query").pack()
knn_query = tk.Entry(knn_frame)
knn_query.pack()

tk.Label(knn_frame, text="K").pack()
knn_k = tk.Entry(knn_frame)
knn_k.pack()

knn_buttons_frame = tk.Frame(knn_frame)
knn_buttons_frame.pack()

tk.Button(knn_buttons_frame, text="Mtree", command=knn_mtree_cb).pack(side=tk.LEFT, padx=4, pady=4)
tk.Button(knn_buttons_frame, text="Sequential", command=knn_sequential_cb).pack(side=tk.RIGHT, padx=4, pady=4)

result_frame = tk.Frame(root, borderwidth=1, relief=tk.RAISED)
result_frame.pack(fill=tk.BOTH, expand=True)

tk.Label(result_frame, text="Result").pack()
result_output = tk.Text(result_frame)
result_output.bind("<Key>", lambda x: "break")
result_output.pack(fill=tk.BOTH, expand=True)


def run():
    create_menu(root)
    update_ui()
    messagebox.showinfo(message="Load or generate mtree")
    root.mainloop()