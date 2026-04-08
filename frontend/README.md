# React + TypeScript + Vite

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Oxc](https://oxc.rs)
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/)

## React Compiler

The React Compiler is not enabled on this template because of its impact on dev & build performances. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

## Expanding the ESLint configuration

If you are developing a production application, we recommend updating the configuration to enable type-aware lint rules:

```js
export default defineConfig([
  # Enterprise RAG Assistant Frontend

  React + TypeScript + Vite UI for the Enterprise RAG Assistant.

  ## What it does

  - Upload PDF documents to the backend knowledge base.
  - Ask questions and view grounded answers.
  - Display cited source documents and page numbers.
  - Show upload and query status inline in the chat UI.

  ## Run locally

  ```bash
  npm install
  npm run dev
  ```

  The app expects the backend API at `http://localhost:8000`.

  ## Notes

  - The current UI is a single-page chat experience with a file upload sidebar.
  - Answers are rendered with Markdown support.
  - The frontend currently consumes the backend `/upload` and `/query` endpoints directly.

  ## Future improvements

  - Environment-based API URL configuration.
  - Conversation history and persistence.
  - Document management and delete actions.
  - Responsive/mobile layout improvements.

