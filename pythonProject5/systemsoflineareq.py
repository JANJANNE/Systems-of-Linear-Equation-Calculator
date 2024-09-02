import streamlit as st
import numpy as np
from linear_equation_solver import lu_decomposition

background_image = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background-image: url("https://png.pngtree.com/thumb_back/fw800/back_our/20190620/ourmid/pngtree-big-benefit-war-car-promotion-poster-background-material-image_150062.jpg");
    background-size: 100vw 100vh;
    background-position: center;  
    background-repeat: repeat;
}
</style>
"""

st.markdown(background_image, unsafe_allow_html=True)

def solve_lu_decomposition(L, U, B):
    n = len(B)
    y = np.zeros(n)
    x = np.zeros(n)

    # Solve Ly = B
    for i in range(n):
        y[i] = B[i] - np.dot(L[i, :i], y[:i])

    # Solve Ux = y
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]

    return x


def main():
    st.markdown(
        "<h1 style='text-align: center; font-size: 36px; font-weight: bold; color: blue ;'>🧩SOLVER APP: UNLOCKING EQUATIONS WITH LU DECOMPOSITION🔍</h1>",
        unsafe_allow_html=True)
    st.markdown(
        "This app solves a system of linear equations using LU decomposition. "
        "Enter the coefficients and the constants to get the solution."
    )

    num_equations = st.number_input("Enter the number of equations:", min_value=2, value=2)

    A = np.zeros((num_equations, num_equations))
    B = np.zeros(num_equations)

    st.subheader("Enter Coefficients in each Equations:")
    equation_columns = st.columns(num_equations)
    for i in range(num_equations):
        with equation_columns[i]:
            st.subheader(f"Equation {i + 1}")
            for j in range(num_equations):
                coeff_title = f"Coefficient {i + 1},{j + 1}"
                A[i, j] = st.number_input(coeff_title, format="%f", key=f"A_{i}_{j}")

    st.subheader("Enter Constants:")
    for i in range(num_equations):
        const_title = f"Constant {i + 1}"
        B[i] = st.number_input(const_title, format="%f", key=f"b_{i}")

    if st.button("Solve"):
        L, U = lu_decomposition(A)
        roots = solve_lu_decomposition(L, U, B)

        st.markdown("""
                            <h1 style="font-size: 25px; font-family: 'Arial, sans-serif'; text-align: center; color: Blue">
                            SHOWING THE STEP-BY-STEP SOLUTION:
                            </h1>
                            """, unsafe_allow_html=True)

        st.markdown("### Lower Triangular Matrix (L):")
        st.write(np.round(L, 3))

        st.markdown("### Upper Triangular Matrix (U):")
        st.write(np.round(U, 3))

        st.markdown("### Solution:")
        for i, root in enumerate(roots):
            st.write(f"X_{i + 1} = {root}")


if __name__ == "__main__":
    main()