import streamlit as st
import plotly.graph_objects as go
import numpy as np
from fractions import Fraction


def create_3d_mesh(fig, vertices, color):
    """
    Create a 3D Mesh trace from given vertices, assuming the face is a pentagon.
    """
    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]

    # Define three triangles for the pentagonal face
    i = [0, 0, 0]  # First vertex
    j = [1, 2, 3]  # Next vertices forming the triangles
    k = [2, 3, 4]  # Closing the triangles

    fig.add_trace(go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,  # Triangles formed from pentagon vertices
        color=color,
        opacity=1.0,
        showscale=False
    ))


def add_side_meshes(fig, vertices1, vertices2, color):
    """
    Add side meshes connecting two sets of vertices.
    """
    n = len(vertices1)
    for idx1 in range(n):
        idx2 = (idx1 + 1) % n  # Loop back to the start

        # Ensure distinct and non-overlapping vertices for each side
        if np.all(vertices1[idx1] == vertices1[idx2]) or np.all(vertices2[idx1] == vertices2[idx2]):
            continue  # Skip degenerate faces

        # Define the vertices for one side (forming a quadrilateral)
        side_vertices = np.array([
            vertices1[idx1],
            vertices1[idx2],
            vertices2[idx2],
            vertices2[idx1],
            vertices1[idx1],
        ])

        # Create a mesh for this side
        create_3d_mesh(fig, side_vertices, color)


def create_3d_object(fig, vertices, thickness, color):
    """
    Create a filled 3D object using Mesh3d.
    """
    vertices2 = vertices + thickness
    create_3d_mesh(fig, vertices, color)
    create_3d_mesh(fig, vertices2, color)
    add_side_meshes(fig, vertices, vertices2, color)


def decimal_to_fraction(value, max_denominator=8):
    """
    Converts a decimal number to a mixed fraction (e.g., 1.125 -> 1 1/8).
    max_denominator limits the denominator's size.
    """
    # Separate the integer and fractional parts
    integer_part = int(value)
    fractional_part = value - integer_part

    # Convert the fractional part to a fraction with the specified max denominator
    fraction = Fraction(fractional_part).limit_denominator(max_denominator)

    if fraction.numerator == 0:
        return f"{integer_part}"  # No fractional part, return the integer part

    if integer_part == 0:
        return f"{fraction}"  # No integer part, return just the fraction

    return f"{integer_part} {fraction}"  # Return mixed number


def annotate_dimension(x0, y0, v):
    fig.add_annotation(
        x=x0, y=y0,
        text=f'{decimal_to_fraction(v) if use_fraction else v}"',
        showarrow=False,
        font=dict(size=12, color="lightgray"),
    )
    xbuff = 0.75
    fig.add_trace(
        go.Scatter(x=[x0 - v / 2, x0 - v / 2], y=[y0 - 0.25, y0 + 0.25], mode='lines', line=dict(color='lightgray')))
    fig.add_trace(
        go.Scatter(x=[x0 + v / 2, x0 + v / 2], y=[y0 - 0.25, y0 + 0.25], mode='lines', line=dict(color='lightgray')))
    fig.add_trace(
        go.Scatter(x=[x0 - v/2, x0 - xbuff], y=[y0, y0], mode='lines', line=dict(color='lightgray')))
    fig.add_trace(
        go.Scatter(x=[x0 + v / 2, x0 + xbuff], y=[y0, y0], mode='lines', line=dict(color='lightgray')))


def calculate_roof_piece1(fig, xyz0, xyz1, l, w, cross_direction):
    # Define the common points p1 and the second point (p0 or p2 depending on the use case)
    p1 = np.array(xyz0)
    p_other = np.array(xyz1)

    # Calculate the direction vector and normalize it
    v1 = p_other - p1
    k = v1 / np.linalg.norm(v1)

    # Calculate the next points p3 (or p6), p4 (or p7), and p5 (or p8)
    p3 = p1 + k * l
    j = np.cross(np.array([0, 0, cross_direction]), k)
    p4 = p3 + w * j
    p5 = p1 + w * j

    # Return the x and y coordinates for the roof piece
    roof_piece_x = [p1[0], p3[0], p4[0], p5[0], p1[0]]
    roof_piece_y = [p1[1], p3[1], p4[1], p5[1], p1[1]]

    fig.add_trace(
        go.Scatter(x=roof_piece_x, y=roof_piece_y, mode='lines', line=dict(color='red')))

    return [p1, p3, p4, p5, p1]


def plot_sheet(fig, sheet_x, sheet_y, shapes, colors=None):
    default_color = 'lightgray'
    if colors is None:
        colors = [default_color]

    iter_colors = iter(colors)

    fig.add_trace(
        go.Scatter(x=baking_sheet_x, y=baking_sheet_y, mode='lines', name='Baking Sheet',
                   line=dict(color=next(iter_colors, default_color))))

    for shape in shapes:
        shape_x, shape_y = shape
        fig.add_trace(go.Scatter(x=shape_x, y=shape_y, mode='lines', line=dict(color=next(iter_colors, default_color))))
    return


st.set_page_config(layout="wide")

with st.sidebar:
    # Streamlit app layout
    st.title('Interactive Gingerbread House Plot')

    # Sliders for user input
    k1 = st.slider('Roof-bottom overhang ratio (k1):', 0.0, 1.0, 0.4, step=0.01)
    k2 = st.slider('Roof-front/back overhang ratio (k2):', 0.0, 1.0, 0.0, step=0.01)

    st.write('Sheet information')
    w = float(st.text_input('Sheet width:', value=14.75))
    h = float(st.text_input('Sheet height:', value=10.25))
    t = float(st.text_input('Thickness:', value=0.5))
    res = float(st.text_input('Cut resolution:', value=1/8))
    use_fraction = st.checkbox('Use fractions:', value=True)

# Derived values
y1 = h / 4
y2 = h / 2
x1_ideal = (y2 + 2 * t) * (1 + 2 * k2)
x1 = round(x1_ideal/res)*res

r_ideal = y2 / (1 + k1)
x3_ideal = (r_ideal**2 - y1**2)**0.5
x2_ideal = (w - x1 - x3_ideal) / 2
x2 = round(x2_ideal/res)*res
x3 = w - x1 - 2*x2
r = (y1**2 + x3**2)**0.5

c1, c2, c3 = st.columns(3)

# Create subplot layout for side-by-side plots
fig = go.Figure()

# Baking sheet
baking_sheet_x = [0, w, w, 0, 0]
baking_sheet_y = [0, 0, h, h, 0]

roof1_x = [0, x1, x1, 0, 0]
roof1_y = [0, 0, y2, y2, 0]
roof2_x = [0, x1, x1, 0, 0]
roof2_y = [y2, y2, h, h, y2]
wall1_x = [x1, x1 + x2, x1 + x2, x1, x1]
wall1_y = [0, 0, y2, y2, 0]
wall2_x = [x1, x1 + x2, x1 + x2, x1, x1]
wall2_y = [y2, y2, h, h, y2]
front_back1_x = [x1 + x2, x1 + 2 * x2, w, x1 + 2 * x2, x1 + x2, x1 + x2]
front_back1_y = [0, 0, y1, y2, y2, 0]
front_back2_x = [x1 + x2, x1 + 2 * x2, w, x1 + 2 * x2, x1 + x2, x1 + x2]
front_back2_y = [y2, y2, y1 + y2, h, h, y2]

shapes = [
    [roof1_x, roof1_y],
    [roof2_x, roof2_y],
    [wall1_x, wall1_y],
    [wall2_x, wall2_y],
    [front_back1_x, front_back1_y],
    [front_back2_x, front_back2_y],
]

colors = ['cyan', 'red', 'red', 'blue', 'blue', 'green', 'green']
plot_sheet(fig, baking_sheet_x, baking_sheet_y, shapes, colors)

# Update layout for side-by-side plots
fig.update_layout(
    xaxis_title='Width',
    yaxis_title='Height',
    showlegend=False,
    xaxis=dict(scaleanchor='y', scaleratio=1, showgrid=True, dtick=1),
    yaxis=dict(scaleanchor='x', scaleratio=1, showgrid=True, dtick=1),
)

with c1:
    # Display the plot
    st.plotly_chart(fig)

fig = go.Figure()
# Front piece of the house (front-on view)
front_x = [0, y2, y2, y1, 0, 0]
front_y = [0, 0, x2, x2 + x3, x2, 0]
fig.add_trace(go.Scatter(x=front_x, y=front_y, mode='lines', name='Front Piece', line=dict(color='green')))

# Roof pieces (front-on view)
# Roof Piece 1
roof1_vertices = calculate_roof_piece1(fig, [y1, x2+x3, 0], [0, x2, 0], y2, t, cross_direction=-1)
roof2_vertices = calculate_roof_piece1(fig, [y1, x2+x3, 0], [y2, x2, 0], y2, t, cross_direction=1)

# Update layout for side-by-side plots
fig.update_layout(
    xaxis_title='Width',
    yaxis_title='Height',
    showlegend=False,
    xaxis=dict(scaleanchor='y', scaleratio=1, showgrid=True, dtick=1),
    yaxis=dict(scaleanchor='x', scaleratio=1, showgrid=True, dtick=1),
)

with c2:
    # Display the plot
    st.plotly_chart(fig)

# 3D Front Piece
fig = go.Figure()
f0, f1, f2, f3, f4, _ = (np.array([x, y, 0]) for x, y in zip(front_x, front_y))

x_thickness = np.array([t, 0, 0])
z_thickness = np.array([0, 0, t])
z_offset_roof = np.array([0, 0, -(x1 - (y2+2*t))/2])

# Create front
front_vertices = np.array([f0, f1, f2, f3, f4, f0])
create_3d_object(fig, front_vertices, z_thickness, 'green')
front_vertices2 = front_vertices + z_thickness

# Create back
back_vertices = front_vertices + z_thickness + np.array([0, 0, y2])
create_3d_object(fig, back_vertices, z_thickness, 'green')

# Create side1
side1_vertices = np.array([front_vertices2[1], front_vertices2[2], back_vertices[2], back_vertices[1], front_vertices2[1]])
create_3d_object(fig, side1_vertices, -x_thickness, 'blue')

# Create side2
side2_vertices = np.array([front_vertices2[4], front_vertices2[5], back_vertices[5], back_vertices[4], front_vertices2[4]])
create_3d_object(fig, side2_vertices, x_thickness, 'blue')

# Create roof1
roof1_vertices3d = np.array([v + z_offset_roof for v in roof1_vertices])
create_3d_object(fig, roof1_vertices3d, np.array([0, 0, x1]), 'red')

# Create roof2
roof2_vertices3d = np.array([v + z_offset_roof for v in roof2_vertices])
create_3d_object(fig, roof2_vertices3d, np.array([0, 0, x1]), 'red')

fig.update_layout(
    scene=dict(
        aspectmode='data',  # Ensure equal scaling along all axes
        xaxis=dict(title='Width', visible=False),
        yaxis=dict(title='Height', visible=False),
        zaxis=dict(title='Depth', visible=False),
        camera=dict(
            up=dict(x=0, y=1, z=0),  # Set the direction of the up vector (use z for vertical)
            eye=dict(x=1, y=.3, z=1.5)  # Adjust x, y, z values to change the default view
        )
    ),
    showlegend=False,
    margin=dict(l=30, r=30, t=50, b=50),
)

with c3:
    # Display the plot
    st.plotly_chart(fig)


# Annotated baking sheet
fig = go.Figure()

plot_sheet(fig, baking_sheet_x, baking_sheet_y, shapes)

x_cutout = [w-x3, w, w-x3, w, w-x3, w, w, w-x3]
y_cutout = [0, y1, y2, h-y1, h, h, 0]

# Plot the area (cutout) using a filled polygon
fig.add_trace(go.Scatter(
    x=x_cutout, y=y_cutout,
    mode='lines', fill='toself',
    line=dict(color='lightgray'),
    fillcolor='rgba(200, 200, 200, 0.5)',  # Light fill color to see hatching
    name="Cutout Area"
))

fig.add_annotation(
    x=w / 2, y=-0.5,  # Position the annotation below the baking sheet
    text=f'Width: {decimal_to_fraction(w) if use_fraction else w}"',
    showarrow=False,
    yshift=-10,
    font=dict(size=12, color="lightgray"),
)

fig.add_annotation(
    x=-0.5, y=h / 2,  # Position the annotation to the left of the baking sheet
    text=f'Height: {decimal_to_fraction(h) if use_fraction else h}"',
    showarrow=False,
    xshift=-10,
    font=dict(size=12, color="lightgray"),
    textangle=-90,  # Rotate the text for vertical display
)

annotate_dimension(x1/2, h+0.75, x1)
annotate_dimension(x1+x2/2, h+0.75, x2)
annotate_dimension(x1+x2+x2/2, h+0.75, x2)

# Update layout for side-by-side plots
fig.update_layout(
    margin=dict(l=40, r=40, t=0, b=50),
    xaxis_title='Width',
    yaxis_title='Height',
    showlegend=False,
    xaxis=dict(visible=False, scaleanchor='y', scaleratio=1),
    yaxis=dict(visible=False, scaleanchor='x', scaleratio=1),
    height=400,
)

# Display the plot
st.plotly_chart(fig)
