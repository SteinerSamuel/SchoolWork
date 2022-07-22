// Main window which will be renderd
import DashboardLayout from "@/pages/Layout/DashboardLayout.vue";
// Children which will be rendered within the main window
import Dashboard from "@/pages/Dashboard.vue";

import TableList from "@/pages/TableList.vue";
import admin from "@/pages/admin.vue";
import AddBook from "@/pages/AddBook.vue";

const routes = [
  {
    path: "/",
    component: DashboardLayout,
    redirect: "/home",
    children: [
      {
        path: "home",
        name: "Home",
        component: Dashboard
      },
      {
        path: "table",
        name: "Search Library",
        component: TableList
      },
      {
        path: "admin",
        name: "Admin",
        component: admin
      },
      {
        path: "new_book",
        name: "Add a New Book",
        component: AddBook
      }
    ]
  }
];

export default routes;
