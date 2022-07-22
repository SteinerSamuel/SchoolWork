<template>
<!-- Start of the dashboard layout which is the parent -->
  <div class="wrapper" :class="{ 'nav-open': $sidebar.showSidebar }"> <!-- The container for the whole webpage -->
  <!-- The left sidebar this includes links to each page which can be loaded -->
    <side-bar>
      <mobile-menu slot="content"></mobile-menu> <!-- Loads the mobile menu if needed --> 
      <!-- list of links to each page first page is the home page -->
      <sidebar-link to="/home"> 
        <md-icon>home</md-icon>
        <p>Home</p>
      </sidebar-link>
      <!-- Link to the table page which the searvh page -->
      <sidebar-link to="/table">
        <md-icon>library_books</md-icon>
        <p>Search</p>
      </sidebar-link>
      <!-- Librarioan/Admin only see these page links  first is a link to the admin pannel-->
      <sidebar-link v-if="$keycloak.hasRealmRole('librarian')" to="/admin">
        <md-icon>person</md-icon>
        <p>Admin</p>
      </sidebar-link>
      <sidebar-link v-if="$keycloak.hasRealmRole('librarian')" to="/new_book">
        <md-icon>library_add</md-icon>
        <p>Add Book</p>
      </sidebar-link>
    </side-bar>

<!-- The main content of the page -->
    <div class="main-panel">
      <!-- loads the top navbar located at ./TopNavbar.vue-->
      <top-navbar></top-navbar>
      <!--Loads the dashboard contetn -->
      <dashboard-content> </dashboard-content>
      <!-- Loads the contetn footer if meta tag is not true -->
      <content-footer v-if="!$route.meta.hideFooter"></content-footer>
    </div>
  </div>
  <!--  End of template -->
</template>


<style lang="scss"></style>
<script>
import TopNavbar from "./TopNavbar.vue";
import ContentFooter from "./ContentFooter.vue";
import DashboardContent from "./Content.vue";
import MobileMenu from "@/pages/Layout/MobileMenu.vue";

export default {
  components: {
    TopNavbar,
    DashboardContent,
    ContentFooter,
    MobileMenu
  }
};
</script>