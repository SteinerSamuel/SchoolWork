<template>
  <div class="content">
    <div v-if="$keycloak.hasRealmRole('librarian')">
      <div class="md-layout">
        <md-card class="md-card-plain">
          <md-card-header data-background-color="blue">
            <h4 class="title">Admin Pannel</h4>
            <p class="category">
              Powered By 
              <a
                href="http://localhost:8090/auth/admin/se_relm/console/#/realms/se_relm/users"
              >
               Keycloak 
              </a>
            </p>
          </md-card-header>

          <md-card-content>
            <div class="iframe-container">
              <iframe
                src="http://localhost:8090/auth/admin/se_relm/console/#/realms/se_relm/users"
              >
              <p>Your browser does not support iframes.</p>
              </iframe>
            </div>
          </md-card-content>
        </md-card>
      </div>
    </div>
    <div v-else>
      <h1>Access Denied</h1>
    </div>
  </div>
</template>

<script>
export default {
  data: () => {}
};
</script>
